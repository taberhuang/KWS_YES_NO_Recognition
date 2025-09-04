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

import { expect } from '@open-wc/testing';
import '../src/components/log-viewer';
import { MockLogSource } from '../src/custom/mock-log-source';
import { createLogViewer } from '../src/createLogViewer';

describe('log-list', () => {
  let mockLogSource;
  let logViewer;

  beforeEach(() => {
    window.localStorage.clear();

    // Initialize the log viewer component with a mock log source
    mockLogSource = new MockLogSource();
    logViewer = createLogViewer(mockLogSource, document.body);

    // Handle benign error caused by custom log viewer initialization
    // See: https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver#observation_errors
    const e = window.onerror;
    window.onerror = function (err) {
      if (
        err === 'ResizeObserver loop completed with undelivered notifications.'
      ) {
        console.warn(
          'Ignored: ResizeObserver loop completed with undelivered notifications.',
        );
        return false;
      } else {
        return e(...arguments);
      }
    };
  });

  afterEach(() => {
    mockLogSource.stop();
    logViewer(); // Destroy the LogViewer instance
  });

  it('displays log entry content correctly', async () => {
    const numLogs = 5;
    const logEntries = [];

    for (let i = 0; i < numLogs; i++) {
      const logEntry = mockLogSource.readLogEntryFromHost();
      logEntries.push(logEntry);
    }

    const logViewer = document.querySelector('log-viewer');
    logViewer.logs = logEntries;

    // Wait for components to load
    await logViewer.updateComplete;
    await new Promise((resolve) => setTimeout(resolve, 200));

    const logView = logViewer.shadowRoot.querySelector('log-view');
    const logList = logView.shadowRoot.querySelector('log-list');
    const table = logList.shadowRoot.querySelector('table');
    const tableRows = table.querySelectorAll('tbody tr');

    // Iterate through log entries and compare each one to its respective <tr> element.
    for (let i = 0; i < numLogs; i++) {
      const logEntry = logEntries[i];
      const logEntryFieldsExceptLevel = logEntry.fields.filter(
        (field) => field.key !== 'level', // Skip `level`
      );
      const tds = tableRows[i].querySelectorAll('td');

      // Iterate through the fields and compare them to the text content of the <td> elements
      for (let j = 0; j < logEntryFieldsExceptLevel.length; j++) {
        const field = logEntryFieldsExceptLevel[j];
        let tdTextContent = tds[j + 1].textContent; // Skip `level`

        // Remove extraneous characters (spaces, newlines, etc.)
        const fieldText = field.value
          .replace(/\n/g, ' ')
          .replace(/ +/g, ' ')
          .trim();
        tdTextContent = tdTextContent
          .replace(/\n/g, ' ')
          .replace(/ +/g, ' ')
          .trim();

        expect(tdTextContent).to.equal(fieldText);
      }
    }
  });
});
