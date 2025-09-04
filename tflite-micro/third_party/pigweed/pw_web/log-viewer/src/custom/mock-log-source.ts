// Copyright 2023 The Pigweed Authors
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import { LogSource } from '../log-source';
import { LogEntry, Level } from '../shared/interfaces';
import { timeFormat } from '../shared/time-format';

export class MockLogSource extends LogSource {
  private intervalId: NodeJS.Timeout | null = null;

  constructor(sourceName = 'Mock Log Source') {
    super(sourceName);
  }

  start(): void {
    const getRandomInterval = () => {
      return Math.floor(Math.random() * (200 - 50 + 1) + 50);
    };

    const readLogEntry = () => {
      const logEntry = this.readLogEntryFromHost();
      this.publishLogEntry(logEntry);

      const nextInterval = getRandomInterval();
      setTimeout(readLogEntry, nextInterval);
    };

    readLogEntry();
  }

  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  getLevel(): Level {
    interface ValueWeightPair {
      level: Level;
      weight: number;
    }

    const valueWeightPairs: ValueWeightPair[] = [
      { level: Level.INFO, weight: 7.45 },
      { level: Level.DEBUG, weight: 0.25 },
      { level: Level.WARNING, weight: 1.5 },
      { level: Level.ERROR, weight: 0.5 },
      { level: Level.CRITICAL, weight: 0.05 },
    ];

    const totalWeight = valueWeightPairs.reduce(
      (acc, pair) => acc + pair.weight,
      0,
    );
    let randomValue = Level.INFO;
    let randomNum = Math.random() * totalWeight;

    for (let i = 0; i < valueWeightPairs.length; i++) {
      randomNum -= valueWeightPairs[i].weight;
      if (randomNum <= 0) {
        randomValue = valueWeightPairs[i].level;
        break;
      }
    }

    return randomValue;
  }

  readLogEntryFromHost(): LogEntry {
    // Emulate reading log data from a host device
    const sources = ['application', 'server', 'database', 'network'];
    const messages = [
      'Request processed successfully!',
      'An unexpected error occurred while performing the operation.',
      'Connection timed out. Please check your network settings.',
      'Invalid input detected. Please provide valid data.',
      'Database connection lost. Attempting to reconnect.',
      'User authentication failed. Invalid credentials provided.',
      'System reboot initiated. Please wait for the system to come back online.',
      'File not found. (The requested file does not exist).',
      'Data corruption detected. Initiating recovery process.',
      'Network congestion detected. Traffic is high, please try again later.',
      'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam condimentum auctor justo, sit amet condimentum nibh facilisis non. Quisque in quam a urna dignissim cursus. Suspendisse egestas nisl sed massa dictum dictum. In tincidunt arcu nec odio eleifend, vel pharetra justo iaculis. Vivamus quis tellus ac velit vehicula consequat. Nam eu felis sed risus hendrerit faucibus ac id lacus. Vestibulum tincidunt tellus in ex feugiat interdum. Nulla sit amet luctus neque. Mauris et aliquet nunc, vel finibus massa. Curabitur laoreet eleifend nibh eget luctus. Fusce sodales augue nec purus faucibus, vel tristique enim vehicula. Aenean eu magna eros. Fusce accumsan dignissim dui auctor scelerisque. Proin ultricies nunc vel tincidunt facilisis.',
    ];
    const level = this.getLevel();
    const timestamp: Date = new Date();
    const getRandomValue = (values: string[]) => {
      const randomIndex = Math.floor(Math.random() * values.length);
      return values[randomIndex];
    };

    const logEntry: LogEntry = {
      level: level,
      timestamp: timestamp,
      fields: [
        { key: 'level', value: level },
        { key: 'time', value: timeFormat.format(timestamp) },
        { key: 'source', value: getRandomValue(sources) },
        { key: 'message', value: getRandomValue(messages) },
      ],
    };

    return logEntry;
  }
}
