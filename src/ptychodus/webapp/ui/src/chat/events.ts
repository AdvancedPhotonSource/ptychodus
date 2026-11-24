/**
 * The AG-UI events the chat panel reads, as the agent service sends them: camelCase JSON,
 * one per SSE `data:` line. Events of any other type are ignored, since the protocol
 * names reasoning events differently from version to version.
 */

export interface RunStarted {
  type: 'RUN_STARTED';
  threadId: string;
  runId: string;
}

export interface RunFinished {
  type: 'RUN_FINISHED';
}

export interface RunError {
  type: 'RUN_ERROR';
  message: string;
}

export interface TextMessageStart {
  type: 'TEXT_MESSAGE_START';
  messageId: string;
}

export interface TextMessageContent {
  type: 'TEXT_MESSAGE_CONTENT';
  messageId: string;
  delta: string;
}

export interface TextMessageEnd {
  type: 'TEXT_MESSAGE_END';
  messageId: string;
}

export interface ToolCallStart {
  type: 'TOOL_CALL_START';
  toolCallId: string;
  toolCallName: string;
}

export interface ToolCallArgs {
  type: 'TOOL_CALL_ARGS';
  toolCallId: string;
  delta: string;
}

export interface ToolCallEnd {
  type: 'TOOL_CALL_END';
  toolCallId: string;
}

export interface ToolCallResult {
  type: 'TOOL_CALL_RESULT';
  toolCallId: string;
  content: string;
}

export interface Custom {
  type: 'CUSTOM';
  name: string;
  value: unknown;
}

export type AgentEvent =
  | RunStarted
  | RunFinished
  | RunError
  | TextMessageStart
  | TextMessageContent
  | TextMessageEnd
  | ToolCallStart
  | ToolCallArgs
  | ToolCallEnd
  | ToolCallResult
  | Custom;

/** The CUSTOM event carrying a text message's HTML, sanitized by the agent service. */
export const RENDERED_TEXT_EVENT = 'ptychodus.rendered_text';

export interface RenderedText {
  messageId: string;
  html: string;
  /** How much of the message's raw text, in UTF-16 code units, `html` renders. */
  sourceLength: number;
  final: boolean;
}

const KNOWN_TYPES = new Set<string>([
  'RUN_STARTED',
  'RUN_FINISHED',
  'RUN_ERROR',
  'TEXT_MESSAGE_START',
  'TEXT_MESSAGE_CONTENT',
  'TEXT_MESSAGE_END',
  'TOOL_CALL_START',
  'TOOL_CALL_ARGS',
  'TOOL_CALL_END',
  'TOOL_CALL_RESULT',
  'CUSTOM',
]);

/** The event a parsed `data:` payload holds, or null for one the panel does not read. */
export function asAgentEvent(payload: unknown): AgentEvent | null {
  if (typeof payload !== 'object' || payload === null) return null;
  const type = (payload as { type?: unknown }).type;
  return typeof type === 'string' && KNOWN_TYPES.has(type) ? (payload as AgentEvent) : null;
}

/** The rendered text a CUSTOM event carries, or null for any other CUSTOM event. */
export function asRenderedText(event: Custom): RenderedText | null {
  if (event.name !== RENDERED_TEXT_EVENT) return null;
  const value = event.value as Partial<RenderedText> | null;
  if (
    value === null ||
    typeof value !== 'object' ||
    typeof value.messageId !== 'string' ||
    typeof value.html !== 'string' ||
    typeof value.sourceLength !== 'number'
  ) {
    return null;
  }
  return { messageId: value.messageId, html: value.html, sourceLength: value.sourceLength, final: !!value.final };
}
