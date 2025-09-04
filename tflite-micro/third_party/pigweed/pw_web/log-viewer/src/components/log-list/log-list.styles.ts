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

import { css } from 'lit';

export const styles = css`
  * {
    box-sizing: border-box;
  }

  :host {
    background-color: var(--sys-log-viewer-color-table-bg);
    color: var(--sys-log-viewer-color-table-text);
    display: block;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.875rem;
    height: 100%;
    overflow: hidden;
    position: relative;
  }

  .table-container {
    display: grid;
    height: 100%;
    overflow: scroll;
    scroll-behavior: auto;
    width: 100%;
  }

  table {
    border-collapse: collapse;
    contain: strict;
    display: table;
    height: 100%;
    table-layout: fixed;
  }

  thead,
  th {
    position: sticky;
    top: 0;
    z-index: 1;
  }

  thead {
    background-color: var(--sys-log-viewer-color-table-header-bg);
    color: var(--sys-log-viewer-color-table-header-text);
    display: block;
    width: 100%;
  }

  tr {
    color: var(--md-sys-color-on-surface);
    border-bottom: 1px solid var(--sys-log-viewer-color-table-cell-outline);
    contain: content;
    display: grid;
    grid-template-columns: var(--column-widths);
    justify-content: flex-start;
    width: 100%;
    will-change: transform, grid-template-columns;
  }

  .log-row--info {
    --icon-color: var(--sys-log-viewer-color-info);
  }

  .log-row--warning {
    --bg-color: var(--sys-log-viewer-color-surface-yellow);
    --text-color: var(--sys-log-viewer-color-on-surface-yellow);
    --icon-color: var(--sys-log-viewer-color-orange-bright);
  }

  .log-row--error,
  .log-row--critical {
    --bg-color: var(--sys-log-viewer-color-surface-error);
    --text-color: var(--sys-log-viewer-color-on-surface-error);
    --icon-color: var(--sys-log-viewer-color-error-bright);
  }

  .log-row--debug {
    --bg-color: initial;
    --text-color: var(--sys-log-viewer-color-debug);
    --icon-color: var(--sys-log-viewer-color-debug);
  }

  .log-row--info .cell-icon,
  .log-row--warning .cell-icon,
  .log-row--error .cell-icon,
  .log-row--critical .cell-icon {
    color: var(--icon-color);
  }

  .log-row--warning,
  .log-row--error,
  .log-row--critical,
  .log-row--debug {
    background-color: var(--bg-color);
    color: var(--text-color);
  }

  .log-row .cell-content {
    display: inline-flex;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: pre-wrap;
  }

  .log-row--nowrap .cell-content {
    white-space: pre;
  }

  tbody tr::before {
    background-color: transparent;
    bottom: 0;
    content: '';
    display: block;
    left: 0;
    position: absolute;
    right: 0;
    top: 0;
    width: 100%;
    z-index: -1;
  }

  tbody tr:hover::before {
    background-color: rgba(
      var(--sys-log-viewer-color-table-row-highlight),
      0.05
    );
  }

  th,
  td {
    display: block;
    grid-row: 1;
    overflow: hidden;
    padding: var(--sys-log-viewer-table-cell-padding);
    text-align: left;
    text-overflow: ellipsis;
  }

  th[hidden],
  td[hidden] {
    display: none;
  }

  th {
    grid-row: 1;
    white-space: nowrap;
  }

  th[title='level'] {
    color: transparent;
  }

  td {
    display: inline-flex;
    position: relative;
    vertical-align: top;
    align-items: flex-start;
  }

  .level-cell {
    align-items: flex-start;
    justify-content: center;
    padding-left: 0;
    padding-right: 0;
  }

  .cell-text {
    line-height: normal;
    text-overflow: ellipsis;
    overflow: hidden;
  }

  .jump-to-bottom-btn {
    bottom: 2.25rem;
    font-family: 'Roboto Flex', sans-serif;
    position: absolute;
    place-self: center;
    transform: translateY(15%) scale(0.9);
  }

  .resize-handle {
    background-color: var(--sys-log-viewer-color-table-cell-outline);
    bottom: 0;
    content: '';
    cursor: col-resize;
    height: 100%;
    mix-blend-mode: luminosity;
    opacity: 1;
    pointer-events: auto;
    position: absolute;
    right: 0;
    top: 0;
    transition: opacity 300ms ease;
    width: 1px;
    z-index: 1;
  }

  .resize-handle:hover {
    background-color: var(--sys-log-viewer-color-primary);
    mix-blend-mode: unset;
    outline: 1px solid var(--sys-log-viewer-color-primary);
  }

  .resize-handle::before {
    bottom: 0;
    content: '';
    display: block;
    position: absolute;
    right: -0.5rem;
    top: 0;
    width: 1rem;
  }

  .cell-icon {
    display: block;
    font-variation-settings:
      'FILL' 1,
      'wght' 400,
      'GRAD' 200,
      'opsz' 58;
    font-size: var(--sys-log-viewer-table-cell-icon-size);
    user-select: none;
    display: grid;
    place-content: center;
    place-items: center;
    width: var(--sys-log-viewer-table-cell-icon-size);
  }

  .overflow-indicator {
    mix-blend-mode: multiply;
    pointer-events: none;
    position: absolute;
    width: 8rem;
  }

  .bottom-indicator {
    align-self: flex-end;
    background: linear-gradient(
      to bottom,
      transparent,
      var(--sys-log-viewer-color-overflow-indicator)
    );
    height: 8rem;
    pointer-events: none;
    position: absolute;
    width: calc(100% - 1rem);
  }

  .left-indicator {
    background: linear-gradient(
      to left,
      transparent,
      var(--sys-log-viewer-color-overflow-indicator)
    );
    height: calc(100% - 1rem);
    justify-self: flex-start;
  }

  .right-indicator {
    background: linear-gradient(
      to right,
      transparent,
      var(--sys-log-viewer-color-overflow-indicator)
    );
    height: calc(100% - 1rem);
    justify-self: flex-end;
  }

  mark {
    background-color: var(--sys-log-viewer-color-table-mark);
    border-radius: 4px;
    color: var(--sys-log-viewer-color-table-mark-text);
  }

  .jump-to-bottom-btn,
  .bottom-indicator {
    opacity: 0;
    transition:
      opacity 100ms ease,
      transform 100ms ease,
      visibility 100ms ease;
    visibility: hidden;
  }

  .jump-to-bottom-btn[data-visible='true'],
  .bottom-indicator[data-visible='true'] {
    opacity: 1;
    transform: translateY(0) scale(1);
    transition:
      opacity 250ms ease,
      transform 250ms ease,
      250ms ease;
    visibility: visible;
  }

  ::-webkit-scrollbar {
    box-shadow: inset 0 0 2rem 2rem var(--md-sys-color-surface-container-low);
    -webkit-appearance: auto;
  }

  ::-webkit-scrollbar-corner {
    background: var(--md-sys-color-surface-container-low);
  }

  ::-webkit-scrollbar-thumb {
    border-radius: 20px;
    box-shadow: inset 0 0 2rem 2rem var(--md-sys-color-outline-variant);
    border: inset 3px transparent;
  }

  ::-webkit-scrollbar-thumb:horizontal {
    border-top: inset 4px transparent;
  }

  ::-webkit-scrollbar-thumb:vertical {
    border-left: inset 4px transparent;
    height: calc(100% / 3);
  }

  ::-webkit-scrollbar-track:horizontal {
    border-top: solid 1px var(--md-sys-color-outline-variant);
  }

  ::-webkit-scrollbar-track:vertical {
    border-left: solid 1px var(--md-sys-color-outline-variant);
  }
`;
