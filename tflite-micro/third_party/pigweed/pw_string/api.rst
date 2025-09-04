.. _module-pw_string-api:

=============
API Reference
=============
.. pigweed-module-subpage::
   :name: pw_string

--------
Overview
--------
This module provides two types of strings and utility functions for working
with strings.

**pw::StringBuilder**

.. doxygenfile:: pw_string/string_builder.h
   :sections: briefdescription

**pw::InlineString**

.. doxygenfile:: pw_string/string.h
   :sections: briefdescription

**String utility functions**

.. doxygenfile:: pw_string/util.h
   :sections: briefdescription

**UTF-8 Helpers**

.. doxygenfile:: pw_string/utf_codecs.h
   :sections: briefdescription

-----------------
pw::StringBuilder
-----------------
.. doxygenfile:: pw_string/string_builder.h
   :sections: briefdescription
.. doxygenclass:: pw::StringBuilder
   :members:

----------------
pw::InlineString
----------------
.. doxygenfile:: pw_string/string.h
   :sections: detaileddescription

.. doxygenclass:: pw::InlineBasicString
   :members:

.. doxygentypedef:: pw::InlineString

------------------------
String utility functions
------------------------

pw::string::Assign()
--------------------
.. doxygenfunction:: pw::string::Assign(InlineString<> &string, std::string_view view)

pw::string::Append()
--------------------
.. doxygenfunction:: pw::string::Append(InlineString<>& string, std::string_view view)

pw::string::ClampedCString()
----------------------------
.. doxygenfunction:: pw::string::ClampedCString(const char* str, size_t max_len)
.. doxygenfunction:: pw::string::ClampedCString(span<const char> str)

pw::string::Copy()
------------------
.. doxygenfunction:: pw::string::Copy(const char* source, char* dest, size_t num)
.. doxygenfunction:: pw::string::Copy(const char* source, Span&& dest)
.. doxygenfunction:: pw::string::Copy(std::string_view source, Span&& dest)

It also has variants that provide a destination of ``pw::Vector<char>``
(see :ref:`module-pw_containers` for details) that do not store the null
terminator in the vector.

.. cpp:function:: StatusWithSize Copy(std::string_view source, pw::Vector<char>& dest)
.. cpp:function:: StatusWithSize Copy(const char* source, pw::Vector<char>& dest)

pw::string::Format()
--------------------
.. doxygenfile:: pw_string/format.h
   :sections: detaileddescription

.. doxygenfunction:: pw::string::Format(span<char> buffer, const char* format, ...)
.. doxygenfunction:: pw::string::FormatVaList(span<char> buffer, const char* format, va_list args)
.. doxygenfunction:: pw::string::Format(InlineString<>& string, const char* format, ...)
.. doxygenfunction:: pw::string::FormatVaList(InlineString<>& string, const char* format, va_list args)
.. doxygenfunction:: pw::string::FormatOverwrite(InlineString<>& string, const char* format, ...)
.. doxygenfunction:: pw::string::FormatOverwriteVaList(InlineString<>& string, const char* format, va_list args)

pw::string::NullTerminatedLength()
----------------------------------
.. doxygenfunction:: pw::string::NullTerminatedLength(const char* str, size_t max_len)
.. doxygenfunction:: pw::string::NullTerminatedLength(span<const char> str)

pw::string::PrintableCopy()
---------------------------
.. doxygenfunction:: pw::string::PrintableCopy(std::string_view source, span<char> dest)

-------------
UTF-8 Helpers
-------------
.. doxygenfile:: pw_string/utf_codecs.h
   :sections: detaileddescription

.. doxygenfunction:: pw::utf8::EncodeCodePoint(uint32_t code_point)
.. doxygenfunction:: pw::utf8::WriteCodePoint(uint32_t code_point, pw::StringBuilder& output)
.. doxygenfunction:: pw::utf8::ReadCodePoint(std::string_view str)
