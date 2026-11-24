/**
 * The agent service's client: its JSON routes under /agent/v1 on the webapp's origin, and
 * each turn as an AG-UI event stream over SSE. Mirrors ptychodus/agent/schemas.py.
 */

import { asAgentEvent, type AgentEvent } from './events.js';

const AGENT = '/agent/v1';

export interface SessionInfo {
  user: string;
  provider: string;
  modelId: string;
}

export interface ConversationInfo {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
}

export interface UserItem {
  kind: 'user';
  text: string;
}

export interface AssistantItem {
  kind: 'assistant';
  /** Sanitized by the agent service. */
  html: string;
}

export interface ToolCallItem {
  kind: 'tool_call';
  toolCallId: string;
  name: string;
  args: Record<string, unknown>;
  ok: boolean | null;
}

export type TranscriptItem = UserItem | AssistantItem | ToolCallItem;

export interface ConversationDetail {
  conversation: ConversationInfo;
  items: TranscriptItem[];
}

/** An answer the agent service refused, with its reason. */
export class AgentError extends Error {
  constructor(
    readonly status: number,
    message: string
  ) {
    super(message);
  }
}

async function send(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<Response> {
  const init: RequestInit = { method, headers: { accept: 'application/json' } };
  if (signal) init.signal = signal;
  if (body !== undefined) {
    init.headers = { accept: 'application/json, text/event-stream', 'content-type': 'application/json' };
    init.body = JSON.stringify(body);
  }
  const res = await fetch(`${AGENT}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const parsed = (await res.json()) as { detail?: unknown };
      if (typeof parsed.detail === 'string') detail = parsed.detail;
    } catch {
      // Not JSON; the status text stands.
    }
    throw new AgentError(res.status, detail);
  }
  return res;
}

async function getJson<T>(path: string): Promise<T> {
  return (await (await send('GET', path)).json()) as T;
}

/** The AG-UI events of an SSE body, as each arrives. */
async function* readEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<AgentEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    for (;;) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let boundary = buffer.indexOf('\n\n');

      while (boundary >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = block
          .split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart())
          .join('\n');
        if (data) {
          const event = asAgentEvent(JSON.parse(data));
          if (event) yield event;
        }
        boundary = buffer.indexOf('\n\n');
      }

      if (done) return;
    }
  } finally {
    reader.releaseLock();
  }
}

export const agentApi = {
  session: () => getJson<SessionInfo>('/session'),
  listConversations: () => getJson<ConversationInfo[]>('/conversations'),
  createConversation: async () =>
    (await (await send('POST', '/conversations', {})).json()) as ConversationInfo,
  getConversation: (id: string) =>
    getJson<ConversationDetail>(`/conversations/${encodeURIComponent(id)}`),
  deleteConversation: async (id: string) => {
    await send('DELETE', `/conversations/${encodeURIComponent(id)}`);
  },

  /** Runs one turn of conversation `id` on `prompt`, yielding its events as they arrive. */
  async *turn(id: string, prompt: string, signal: AbortSignal): AsyncGenerator<AgentEvent> {
    const messageId = crypto.randomUUID();
    const input = {
      threadId: id,
      runId: crypto.randomUUID(),
      messages: [{ id: messageId, role: 'user', content: prompt }],
      tools: [],
      context: [],
    };
    const res = await send('POST', `/conversations/${encodeURIComponent(id)}/turns`, input, signal);
    if (!res.body) throw new AgentError(res.status, 'The agent service sent no event stream.');
    yield* readEvents(res.body);
  },
};
