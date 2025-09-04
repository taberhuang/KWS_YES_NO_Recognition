# Copyright 2020 The Pigweed Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
"""Builds and manages databases of tokenized strings."""

from __future__ import annotations

from abc import abstractmethod
import bisect
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    ValuesView,
)
import csv
from dataclasses import dataclass
from datetime import datetime
import io
import logging
from pathlib import Path
import re
import struct
import subprocess
from typing import (
    Any,
    BinaryIO,
    IO,
    NamedTuple,
    overload,
    Pattern,
    TextIO,
    TypeVar,
)
from uuid import uuid4

DEFAULT_DOMAIN = ''

# The default hash length to use for C-style hashes. This value only applies
# when manually hashing strings to recreate token calculations in C. The C++
# hash function does not have a maximum length.
#
# This MUST match the default value of PW_TOKENIZER_CFG_C_HASH_LENGTH in
# pw_tokenizer/public/pw_tokenizer/config.h.
DEFAULT_C_HASH_LENGTH = 128

TOKENIZER_HASH_CONSTANT = 65599

_LOG = logging.getLogger('pw_tokenizer')


def _value(char: int | str) -> int:
    return char if isinstance(char, int) else ord(char)


def pw_tokenizer_65599_hash(
    string: str | bytes, *, hash_length: int | None = None
) -> int:
    """Hashes the string with the hash function used to generate tokens in C++.

    This hash function is used calculate tokens from strings in Python. It is
    not used when extracting tokens from an ELF, since the token is stored in
    the ELF as part of tokenization.
    """
    hash_value = len(string)
    coefficient = TOKENIZER_HASH_CONSTANT

    for char in string[:hash_length]:
        hash_value = (hash_value + coefficient * _value(char)) % 2**32
        coefficient = (coefficient * TOKENIZER_HASH_CONSTANT) % 2**32

    return hash_value


def c_hash(
    string: str | bytes, hash_length: int = DEFAULT_C_HASH_LENGTH
) -> int:
    """Hashes the string with the hash function used in C."""
    return pw_tokenizer_65599_hash(string, hash_length=hash_length)


@dataclass(frozen=True, eq=True, order=True)
class _EntryKey:
    """Uniquely refers to an entry."""

    domain: str
    token: int
    string: str


class TokenizedStringEntry:
    """A tokenized string with its metadata."""

    def __init__(
        self,
        token: int,
        string: str,
        domain: str = DEFAULT_DOMAIN,
        date_removed: datetime | None = None,
    ) -> None:
        self._key = _EntryKey(
            ''.join(domain.split()),
            token,
            string,
        )
        self.date_removed = date_removed

    @property
    def token(self) -> int:
        return self._key.token

    @property
    def string(self) -> str:
        return self._key.string

    @property
    def domain(self) -> str:
        return self._key.domain

    def key(self) -> _EntryKey:
        """The key determines uniqueness for a tokenized string."""
        return self._key

    def update_date_removed(self, new_date_removed: datetime | None) -> None:
        """Sets self.date_removed if the other date is newer."""
        # No removal date (None) is treated as the newest date.
        if self.date_removed is None:
            return

        if new_date_removed is None or new_date_removed > self.date_removed:
            self.date_removed = new_date_removed

    def __eq__(self, other: Any) -> bool:
        return (
            self.key() == other.key()
            and self.date_removed == other.date_removed
        )

    def __lt__(self, other: Any) -> bool:
        """Sorts the entry by domain, token, date removed, then string."""
        if self.domain != other.domain:
            return self.domain < other.domain

        if self.token != other.token:
            return self.token < other.token

        # Sort removal dates in reverse, so the most recently removed (or still
        # present) entry appears first.
        if self.date_removed != other.date_removed:
            return (other.date_removed or datetime.max) < (
                self.date_removed or datetime.max
            )

        return self.string < other.string

    def __str__(self) -> str:
        return self.string

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(token=0x{self.token:08x}, '
            f'string={self.string!r}, domain={self.domain!r})'
        )


_TokenToEntries = dict[int, list[TokenizedStringEntry]]
_K = TypeVar('_K')
_V = TypeVar('_V')
_T = TypeVar('_T')


class _TokenDatabaseView(Mapping[_K, _V]):  # pylint: disable=abstract-method
    """Read-only mapping view of a token database.

    Behaves like a read-only version of defaultdict(list).
    """

    def __init__(self, mapping: Mapping[_K, Any]) -> None:
        self._mapping = mapping

    def __contains__(self, key: object) -> bool:
        return key in self._mapping

    @overload
    def get(self, key: _K) -> _V | None:  # pylint: disable=arguments-differ
        ...

    @overload
    def get(self, key: _K, default: _T) -> _V | _T:  # pylint: disable=W0222
        ...

    def get(self, key: _K, default: _T | None = None) -> _V | _T | None:
        return self._mapping.get(key, default)

    def __iter__(self) -> Iterator[_K]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __str__(self) -> str:
        return str(self._mapping)

    def __repr__(self) -> str:
        return repr(self._mapping)


class _TokenMapping(_TokenDatabaseView[int, Sequence[TokenizedStringEntry]]):
    def __getitem__(self, token: int) -> Sequence[TokenizedStringEntry]:
        """Returns strings that match the specified token; may be empty."""
        return self._mapping.get(token, ())  # Empty sequence if no match


class _DomainTokenMapping(_TokenDatabaseView[str, _TokenMapping]):
    def __getitem__(self, domain: str) -> _TokenMapping:
        """Returns the token-to-strings mapping for the specified domain."""
        return _TokenMapping(self._mapping.get(domain, {}))  # Empty if no match


def _add_entry(entries: _TokenToEntries, entry: TokenizedStringEntry) -> None:
    bisect.insort(
        entries.setdefault(entry.token, []),
        entry,
        key=TokenizedStringEntry.key,  # Keep lists of entries sorted by key.
    )


class Database:
    """Database of tokenized strings stored as TokenizedStringEntry objects."""

    def __init__(self, entries: Iterable[TokenizedStringEntry] = ()):
        """Creates a token database."""
        # The database dict stores each unique (token, string) entry.
        self._database: dict[_EntryKey, TokenizedStringEntry] = {}

        # Index by token and domain
        self._token_entries: _TokenToEntries = {}
        self._domain_token_entries: dict[str, _TokenToEntries] = {}

        self.add(entries)

    @classmethod
    def from_strings(
        cls,
        strings: Iterable[str],
        domain: str = DEFAULT_DOMAIN,
        tokenize: Callable[[str], int] = pw_tokenizer_65599_hash,
    ) -> Database:
        """Creates a Database from an iterable of strings."""
        return cls(
            TokenizedStringEntry(tokenize(string), string, domain)
            for string in strings
        )

    @classmethod
    def merged(cls, *databases: Database) -> Database:
        """Creates a TokenDatabase from one or more other databases."""
        db = cls()
        db.merge(*databases)
        return db

    @property
    def token_to_entries(self) -> Mapping[int, Sequence[TokenizedStringEntry]]:
        """Returns a mapping of tokens to a sequence of TokenizedStringEntry.

        Returns token database entries from all domains.
        """
        return _TokenMapping(self._token_entries)

    @property
    def domains(
        self,
    ) -> Mapping[str, Mapping[int, Sequence[TokenizedStringEntry]]]:
        """Returns a mapping of domains to tokens to a sequence of entries.

        `database.domains[domain][token]` returns a sequence of strings matching
        the token in the domain, or an empty sequence if there are no matches.
        """
        return _DomainTokenMapping(self._domain_token_entries)

    def entries(self) -> ValuesView[TokenizedStringEntry]:
        """Returns iterable over all TokenizedStringEntries in the database."""
        return self._database.values()

    def collisions(
        self,
    ) -> Iterator[tuple[int, Sequence[TokenizedStringEntry]]]:
        """Returns tuple of (token, entries_list)) for all colliding tokens."""
        for token, entries in self.token_to_entries.items():
            if len(entries) > 1:
                yield token, entries

    def mark_removed(
        self,
        all_entries: Iterable[TokenizedStringEntry],
        removal_date: datetime | None = None,
    ) -> list[TokenizedStringEntry]:
        """Marks entries missing from all_entries as having been removed.

        The entries are assumed to represent the complete set of entries for the
        database. Entries currently in the database not present in the provided
        entries are marked with a removal date but remain in the database.
        Entries in all_entries missing from the database are NOT added; call the
        add function to add these.

        Args:
          all_entries: the complete set of strings present in the database
          removal_date: the datetime for removed entries; today by default

        Returns:
          A list of entries marked as removed.
        """
        if removal_date is None:
            removal_date = datetime.now()

        all_keys = frozenset(entry.key() for entry in all_entries)

        removed = []

        for entry in self._database.values():
            if entry.key() not in all_keys and (
                entry.date_removed is None or removal_date < entry.date_removed
            ):
                # Add a removal date, or update it to the oldest date.
                entry.date_removed = removal_date
                removed.append(entry)

        return removed

    def add(self, entries: Iterable[TokenizedStringEntry]) -> None:
        """Adds new entries and updates date_removed for existing entries.

        If the added tokens have removal dates, the newest date is used.
        """
        for new_entry in entries:
            # Update an existing entry or create a new one.
            try:
                entry = self._database[new_entry.key()]

                # Keep the latest removal date between the two entries.
                if new_entry.date_removed is None:
                    entry.date_removed = None
                elif (
                    entry.date_removed
                    and entry.date_removed < new_entry.date_removed
                ):
                    entry.date_removed = new_entry.date_removed
            except KeyError:
                self._add_new_entry(new_entry)

    def purge(
        self, date_removed_cutoff: datetime | None = None
    ) -> list[TokenizedStringEntry]:
        """Removes and returns entries removed on/before date_removed_cutoff."""
        if date_removed_cutoff is None:
            date_removed_cutoff = datetime.max

        to_delete = [
            entry
            for entry in self._database.values()
            if entry.date_removed and entry.date_removed <= date_removed_cutoff
        ]

        for entry in to_delete:
            self._delete_entry(entry)

        return to_delete

    def merge(self, *databases: Database) -> None:
        """Merges two or more databases together.

        All entries are kept if there are token collisions.

        If there are two identical tokens (same domain, token, and string), the
        newest removal date (or no date if either token has no removal date) is
        used for the merged token.
        """
        for other_db in databases:
            for entry in other_db.entries():
                key = entry.key()

                if key in self._database:
                    self._database[key].update_date_removed(entry.date_removed)
                else:
                    self._add_new_entry(entry)

    def filter(
        self,
        include: Iterable[str | Pattern[str]] = (),
        exclude: Iterable[str | Pattern[str]] = (),
        replace: Iterable[tuple[str | Pattern[str], str]] = (),
    ) -> None:
        """Filters the database using regular expressions (strings or compiled).

        Args:
          include: regexes; only entries matching at least one are kept
          exclude: regexes; entries matching any of these are removed
          replace: (regex, str) tuples; replaces matching terms in all entries
        """
        to_delete: list[TokenizedStringEntry] = []

        if include:
            include_re = [re.compile(pattern) for pattern in include]
            to_delete.extend(
                val
                for val in self._database.values()
                if not any(rgx.search(val.string) for rgx in include_re)
            )

        if exclude:
            exclude_re = [re.compile(pattern) for pattern in exclude]
            to_delete.extend(
                val
                for val in self._database.values()
                if any(rgx.search(val.string) for rgx in exclude_re)
            )

        for entry in to_delete:
            self._delete_entry(entry)

        # Do the replacement after removing entries.
        for search, replacement in replace:
            search = re.compile(search)

            to_replace: list[TokenizedStringEntry] = []
            add: list[TokenizedStringEntry] = []

            for entry in self._database.values():
                new_string = search.sub(replacement, entry.string)
                if new_string != entry.string:
                    to_replace.append(entry)
                    add.append(
                        TokenizedStringEntry(
                            entry.token,
                            new_string,
                            entry.domain,
                            entry.date_removed,
                        )
                    )

            for entry in to_replace:
                self._delete_entry(entry)
            self.add(add)

    def difference(self, other: Database) -> Database:
        """Returns a new Database with entries in this DB not in the other."""
        # pylint: disable=protected-access
        return Database(
            e for k, e in self._database.items() if k not in other._database
        )
        # pylint: enable=protected-access

    def _add_new_entry(self, new_entry: TokenizedStringEntry) -> None:
        entry = TokenizedStringEntry(  # These are mutable, so make a copy.
            new_entry.token,
            new_entry.string,
            new_entry.domain,
            new_entry.date_removed,
        )
        self._database[entry.key()] = entry
        _add_entry(self._token_entries, entry)
        _add_entry(
            self._domain_token_entries.setdefault(entry.domain, {}), entry
        )

    def _delete_entry(self, entry: TokenizedStringEntry) -> None:
        del self._database[entry.key()]

        # Remove from the token / domain mappings and clean up empty lists.
        self._token_entries[entry.token].remove(entry)
        if not self._token_entries[entry.token]:
            del self._token_entries[entry.token]

        self._domain_token_entries[entry.domain][entry.token].remove(entry)
        if not self._domain_token_entries[entry.domain][entry.token]:
            del self._domain_token_entries[entry.domain][entry.token]
            if not self._domain_token_entries[entry.domain]:
                del self._domain_token_entries[entry.domain]

    def __len__(self) -> int:
        """Returns the number of entries in the database."""
        return len(self.entries())

    def __bool__(self) -> bool:
        """True if the database is non-empty."""
        return bool(self._database)

    def __str__(self) -> str:
        """Outputs the database as CSV."""
        csv_output = io.BytesIO()
        write_csv(self, csv_output)
        return csv_output.getvalue().decode()


def parse_csv(fd: TextIO) -> Iterable[TokenizedStringEntry]:
    """Parses TokenizedStringEntries from a CSV token database file."""
    for line in csv.reader(fd):
        try:
            try:
                token_str, date_str, domain, string_literal = line
            except ValueError:
                # If there are only three columns, use the default domain.
                token_str, date_str, string_literal = line
                domain = DEFAULT_DOMAIN

            token = int(token_str, 16)
            date = (
                datetime.fromisoformat(date_str) if date_str.strip() else None
            )

            yield TokenizedStringEntry(token, string_literal, domain, date)
        except (ValueError, UnicodeDecodeError) as err:
            _LOG.error(
                'Failed to parse tokenized string entry %s: %s', line, err
            )


def write_csv(database: Database, fd: IO[bytes]) -> None:
    """Writes the database as CSV to the provided binary file."""
    for entry in sorted(database.entries()):
        _write_csv_line(fd, entry)


def _write_csv_line(fd: IO[bytes], entry: TokenizedStringEntry):
    """Write a line in CSV format to the provided binary file."""
    # Align the CSV output to 10-character columns for improved readability.
    # Use \n instead of RFC 4180's \r\n.
    fd.write(
        '{:08x},{:10},"{}","{}"\n'.format(
            entry.token,
            entry.date_removed.date().isoformat() if entry.date_removed else '',
            entry.domain.replace('"', '""'),  # escape " as ""
            entry.string.replace('"', '""'),
        ).encode()
    )


class _BinaryFileFormat(NamedTuple):
    """Attributes of the binary token database file format."""

    magic: bytes = b'TOKENS\0\0'
    header: struct.Struct = struct.Struct('<8sI4x')
    entry: struct.Struct = struct.Struct('<IBBH')


BINARY_FORMAT = _BinaryFileFormat()


class DatabaseFormatError(Exception):
    """Failed to parse a token database file."""


def file_is_binary_database(fd: BinaryIO) -> bool:
    """True if the file starts with the binary token database magic string."""
    try:
        fd.seek(0)
        magic = fd.read(len(BINARY_FORMAT.magic))
        fd.seek(0)
        return BINARY_FORMAT.magic == magic
    except IOError:
        return False


def _check_that_file_is_csv_database(path: Path) -> None:
    """Raises an error unless the path appears to be a CSV token database."""
    try:
        with path.open('rb') as fd:
            data = fd.read(8)  # Read 8 bytes, which should be the first token.

        if not data:
            return  # File is empty, which is valid CSV.

        if len(data) != 8:
            raise DatabaseFormatError(
                f'Attempted to read {path} as a CSV token database, but the '
                f'file is too short ({len(data)} B)'
            )

        # Make sure the first 8 chars are a valid hexadecimal number.
        _ = int(data.decode(), 16)
    except (IOError, UnicodeDecodeError, ValueError) as err:
        raise DatabaseFormatError(
            f'Encountered error while reading {path} as a CSV token database'
        ) from err


def parse_binary(fd: BinaryIO) -> Iterable[TokenizedStringEntry]:
    """Parses TokenizedStringEntries from a binary token database file."""
    magic, entry_count = BINARY_FORMAT.header.unpack(
        fd.read(BINARY_FORMAT.header.size)
    )

    if magic != BINARY_FORMAT.magic:
        raise DatabaseFormatError(
            f'Binary token database magic number mismatch (found {magic!r}, '
            f'expected {BINARY_FORMAT.magic!r}) while reading from {fd}'
        )

    entries = []

    for _ in range(entry_count):
        token, day, month, year = BINARY_FORMAT.entry.unpack(
            fd.read(BINARY_FORMAT.entry.size)
        )

        try:
            date_removed: datetime | None = datetime(year, month, day)
        except ValueError:
            date_removed = None

        entries.append((token, date_removed))

    # Read the entire string table and define a function for looking up strings.
    string_table = fd.read()

    def read_string(start):
        end = string_table.find(b'\0', start)
        return (
            string_table[start : string_table.find(b'\0', start)].decode(),
            end + 1,
        )

    offset = 0
    for token, removed in entries:
        string, offset = read_string(offset)
        yield TokenizedStringEntry(token, string, DEFAULT_DOMAIN, removed)


def write_binary(database: Database, fd: BinaryIO) -> None:
    """Writes the database as packed binary to the provided binary file."""
    entries = sorted(database.entries())

    fd.write(BINARY_FORMAT.header.pack(BINARY_FORMAT.magic, len(entries)))

    string_table = bytearray()

    for entry in entries:
        if entry.date_removed:
            removed_day = entry.date_removed.day
            removed_month = entry.date_removed.month
            removed_year = entry.date_removed.year
        else:
            # If there is no removal date, use the special value 0xffffffff for
            # the day/month/year. That ensures that still-present tokens appear
            # as the newest tokens when sorted by removal date.
            removed_day = 0xFF
            removed_month = 0xFF
            removed_year = 0xFFFF

        string_table += entry.string.encode()
        string_table.append(0)

        fd.write(
            BINARY_FORMAT.entry.pack(
                entry.token, removed_day, removed_month, removed_year
            )
        )

    fd.write(string_table)


class _ElfFileFormat(NamedTuple):
    """Attributes of the elf token database file format."""

    # Magic number used to indicate the beginning of a tokenized string entry.
    # This value MUST match the value of _PW_TOKENIZER_ENTRY_MAGIC in
    # pw_tokenizer/public/pw_tokenizer/internal/tokenize_string.h.
    magic: int = 0xBAA98DEE
    header: struct.Struct = struct.Struct('<4I')


ELF_FORMAT = _ElfFileFormat()


def write_elf_db_format(database: Database, fd: BinaryIO) -> None:
    """Writes the database as elf file format to the provided binary file."""

    for entry in sorted(database.entries()):
        # The token elf database format requires null-terminated strings.
        # Add a null terminator to the domain and string data.
        domain_data = entry.domain.encode() + b"\0"

        string_data = entry.string.encode() + b"\0"

        # Pack the header
        header_data = ELF_FORMAT.header.pack(
            ELF_FORMAT.magic,
            entry.token,
            len(domain_data),
            len(string_data),
        )

        fd.write(header_data)
        fd.write(domain_data)
        fd.write(string_data)


class DatabaseFile(Database):
    """A token database that is associated with a particular file.

    This class adds the write_to_file() method that writes to file from which it
    was created in the correct format (CSV or binary).
    """

    def __init__(
        self, path: Path, entries: Iterable[TokenizedStringEntry]
    ) -> None:
        super().__init__(entries)
        self.path = path

    @staticmethod
    def load(path: Path) -> DatabaseFile:
        """Creates a DatabaseFile that coincides to the file type."""
        if path.is_dir():
            return _DirectoryDatabase(path)

        # Read the path as a packed binary file.
        with path.open('rb') as fd:
            if file_is_binary_database(fd):
                return _BinaryDatabase(path, fd)

        # Read the path as a CSV file.
        _check_that_file_is_csv_database(path)
        return _CSVDatabase(path)

    @abstractmethod
    def write_to_file(self, *, rewrite: bool = False) -> None:
        """Exports in the original format to the original path."""

    @abstractmethod
    def add_and_discard_temporary(
        self, entries: Iterable[TokenizedStringEntry], commit: str
    ) -> None:
        """Discards and adds entries to export in the original format.

        Adds entries after removing temporary entries from the Database
        to exclusively write re-occurring entries into memory and disk.
        """


class _BinaryDatabase(DatabaseFile):
    def __init__(self, path: Path, fd: BinaryIO) -> None:
        super().__init__(path, parse_binary(fd))

    def write_to_file(self, *, rewrite: bool = False) -> None:
        """Exports in the binary format to the original path."""
        del rewrite  # Binary databases are always rewritten
        with self.path.open('wb') as fd:
            write_binary(self, fd)

    def add_and_discard_temporary(
        self, entries: Iterable[TokenizedStringEntry], commit: str
    ) -> None:
        # TODO: b/241471465 - Implement adding new tokens and removing
        # temporary entries for binary databases.
        raise NotImplementedError(
            '--discard-temporary is currently only '
            'supported for directory databases'
        )


class _CSVDatabase(DatabaseFile):
    def __init__(self, path: Path) -> None:
        with path.open('r', newline='', encoding='utf-8') as csv_fd:
            super().__init__(path, parse_csv(csv_fd))

    def write_to_file(self, *, rewrite: bool = False) -> None:
        """Exports in the CSV format to the original path."""
        del rewrite  # CSV databases are always rewritten
        with self.path.open('wb') as fd:
            write_csv(self, fd)

    def add_and_discard_temporary(
        self, entries: Iterable[TokenizedStringEntry], commit: str
    ) -> None:
        # TODO: b/241471465 - Implement adding new tokens and removing
        # temporary entries for CSV databases.
        raise NotImplementedError(
            '--discard-temporary is currently only '
            'supported for directory databases'
        )


# The suffix used for CSV files in a directory database.
DIR_DB_SUFFIX = '.pw_tokenizer.csv'
DIR_DB_GLOB = '*' + DIR_DB_SUFFIX


def _parse_directory(directory: Path) -> Iterable[TokenizedStringEntry]:
    """Parses TokenizedStringEntries tokenizer CSV files in the directory."""
    for path in directory.glob(DIR_DB_GLOB):
        yield from _CSVDatabase(path).entries()


def _most_recently_modified_file(paths: Iterable[Path]) -> Path:
    return max(paths, key=lambda path: path.stat().st_mtime)


class _DirectoryDatabase(DatabaseFile):
    def __init__(self, directory: Path) -> None:
        super().__init__(directory, _parse_directory(directory))

    def write_to_file(self, *, rewrite: bool = False) -> None:
        """Creates a new CSV file in the directory with any new tokens."""
        if rewrite:
            # Write the entire database to a new CSV file
            new_file = self._create_filename()
            with new_file.open('wb') as fd:
                write_csv(self, fd)

            # Delete all CSV files except for the new CSV with everything.
            for csv_file in self.path.glob(DIR_DB_GLOB):
                if csv_file != new_file:
                    csv_file.unlink()
        else:
            # Reread the tokens from disk and write only the new entries to CSV.
            current_tokens = Database(_parse_directory(self.path))
            new_entries = self.difference(current_tokens)
            if new_entries:
                with self._create_filename().open('wb') as fd:
                    write_csv(new_entries, fd)

    def _git_paths(self, commands: list) -> list[Path]:
        """Returns a list of database CSVs from a Git command."""
        try:
            output = subprocess.run(
                ['git', *commands, DIR_DB_GLOB],
                capture_output=True,
                check=True,
                cwd=self.path,
                text=True,
            ).stdout.strip()
            return [self.path / repo_path for repo_path in output.splitlines()]
        except subprocess.CalledProcessError:
            return []

    def _find_latest_csv(self, commit: str) -> Path:
        """Finds or creates a CSV to which to write new entries.

        - Check for untracked CSVs. Use the most recently modified file, if any.
        - Check for CSVs added in HEAD, if HEAD is not an ancestor of commit.
          Use the most recently modified file, if any.
        - If no untracked or committed files were found, create a new file.
        """

        # Prioritize untracked files in the directory database.
        untracked_changes = self._git_paths(
            ['ls-files', '--others', '--exclude-standard']
        )
        if untracked_changes:
            return _most_recently_modified_file(untracked_changes)

        # Check if HEAD is an ancestor of the base commit. This checks whether
        # the top commit has been merged or not. If it has been merged, create a
        # new CSV to use. Otherwise, check if a CSV was added in the commit.
        head_is_not_merged = (
            subprocess.run(
                ['git', 'merge-base', '--is-ancestor', 'HEAD', commit],
                cwd=self.path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            != 0
        )

        if head_is_not_merged:
            # Find CSVs added in the top commit.
            csvs_from_top_commit = self._git_paths(
                [
                    'diff',
                    '--name-only',
                    '--diff-filter=A',
                    '--relative',
                    'HEAD~',
                ]
            )

            if csvs_from_top_commit:
                return _most_recently_modified_file(csvs_from_top_commit)

        return self._create_filename()

    def _create_filename(self) -> Path:
        """Generates a unique filename not in the directory."""
        # Tracked and untracked files do not exist in the repo.
        while (file := self.path / f'{uuid4().hex}{DIR_DB_SUFFIX}').exists():
            pass
        return file

    def add_and_discard_temporary(
        self, entries: Iterable[TokenizedStringEntry], commit: str
    ) -> None:
        """Adds new entries and discards temporary entries on disk.

        - Find the latest CSV in the directory database or create a new one.
        - Delete entries in the latest CSV that are not in the entries passed to
          this function.
        - Add the new entries to this database.
        - Overwrite the latest CSV with only the newly added entries.
        """
        # Find entries not currently in the database.
        added = Database(entries)
        new_entries = added.difference(self)

        csv_path = self._find_latest_csv(commit)
        if csv_path.exists():
            # Loading the CSV as a DatabaseFile.
            csv_db = DatabaseFile.load(csv_path)

            # Delete entries added in the CSV, but not added in this function.
            for key in (e.key() for e in csv_db.difference(added).entries()):
                del self._database[key]
                del csv_db._database[key]  # pylint: disable=protected-access

            csv_db.add(new_entries.entries())
            csv_db.write_to_file()
        elif new_entries:  # If the CSV does not exist, write all new tokens.
            with csv_path.open('wb') as fd:
                write_csv(new_entries, fd)

        self.add(new_entries.entries())
