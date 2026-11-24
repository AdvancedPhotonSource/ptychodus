/**
 * The chat drawer beside the viewer: the user's conversations with the agent service.
 *
 * Text the user or a tool wrote is set as text. The only HTML inserted is what the agent
 * service rendered and sanitized: a message's `ptychodus.rendered_text` events while a
 * turn streams, and each assistant item of a reloaded conversation. Raw deltas show as
 * text until the HTML of the blocks they complete replaces them.
 */

import { agentApi, AgentError, type ConversationInfo, type TranscriptItem } from './agent_api.js';
import { asRenderedText, type AgentEvent } from './events.js';
import { RENDER_CALLS } from './render_calls.js';

const OPEN_KEY = 'ptychodus.chat.open';
const REM_PX = () => parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;

interface TextMessage {
  rendered: HTMLElement;
  raw: HTMLElement;
  text: string;
  /** The UTF-16 length of the text `rendered` shows. */
  renderedLength: number;
}

interface ToolCall {
  chip: HTMLElement;
  name: string;
  args: string;
}

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function readOpen(): boolean {
  try {
    return window.localStorage.getItem(OPEN_KEY) === 'true';
  } catch {
    return false;
  }
}

function writeOpen(open: boolean): void {
  try {
    window.localStorage.setItem(OPEN_KEY, String(open));
  } catch {
    // Storage is unavailable; the drawer just forgets.
  }
}

/** Opens links that leave the web UI in a new tab, so the conversation stays. */
function retargetLinks(root: HTMLElement): void {
  for (const a of root.querySelectorAll<HTMLAnchorElement>('a[href]')) {
    if (new URL(a.href, window.location.href).origin !== window.location.origin) a.target = '_blank';
  }
}

export class ChatPanel {
  private readonly root: HTMLElement;
  private readonly select: HTMLSelectElement;
  private readonly deleteButton: HTMLButtonElement;
  private readonly modelLabel: HTMLElement;
  private readonly messages: HTMLElement;
  private readonly input: HTMLTextAreaElement;
  private readonly sendButton: HTMLButtonElement;
  private readonly stopButton: HTMLButtonElement;
  private toggle: HTMLButtonElement | null = null;
  private loaded = false;
  private available = false;
  private conversationId: string | null = null;
  private controller: AbortController | null = null;
  private texts = new Map<string, TextMessage>();
  private toolCalls = new Map<string, ToolCall>();

  constructor(root: HTMLElement) {
    this.root = root;
    root.classList.add('chat');
    root.setAttribute('aria-label', 'Agent');

    const resizer = element('div', 'chat-resizer');
    resizer.setAttribute('role', 'separator');
    resizer.setAttribute('aria-orientation', 'vertical');
    resizer.addEventListener('pointerdown', (ev) => this.resize(resizer, ev));

    const header = element('div', 'chat-header');
    this.select = element('select', 'chat-conversations');
    this.select.setAttribute('aria-label', 'Conversation');
    this.select.addEventListener('change', () => void this.openConversation(this.select.value || null));
    const newButton = element('button', 'chat-button', 'New');
    newButton.type = 'button';
    newButton.title = 'Start a new conversation';
    newButton.addEventListener('click', () => void this.openConversation(null));
    this.deleteButton = element('button', 'chat-button', 'Delete');
    this.deleteButton.type = 'button';
    this.deleteButton.title = 'Delete this conversation';
    this.deleteButton.addEventListener('click', () => void this.deleteConversation());
    this.modelLabel = element('div', 'chat-model');
    header.append(this.select, newButton, this.deleteButton, this.modelLabel);

    this.messages = element('div', 'chat-messages');
    this.messages.setAttribute('aria-live', 'polite');

    const form = element('form', 'chat-form');
    this.input = element('textarea', 'chat-input');
    this.input.rows = 3;
    this.input.placeholder = 'Ask about the storage root…';
    this.input.setAttribute('aria-label', 'Message');
    this.input.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) {
        ev.preventDefault();
        void this.send();
      }
    });
    this.sendButton = element('button', 'chat-send');
    this.sendButton.type = 'submit';
    this.sendButton.title = 'Send';
    this.sendButton.setAttribute('aria-label', 'Send');
    const sendIcon = element('img', '');
    sendIcon.src = '/ui/icons/paper-plane.svg';
    sendIcon.alt = '';
    this.sendButton.appendChild(sendIcon);
    this.stopButton = element('button', 'chat-stop', 'Stop');
    this.stopButton.type = 'button';
    this.stopButton.hidden = true;
    this.stopButton.addEventListener('click', () => this.controller?.abort());
    form.append(this.input, this.sendButton, this.stopButton);
    form.addEventListener('submit', (ev) => {
      ev.preventDefault();
      void this.send();
    });

    root.append(resizer, header, this.messages, form);
    this.setOpen(readOpen());
  }

  /** A nav button opening and closing the drawer; the nav builds a new one each render. */
  toggleButton(): HTMLButtonElement {
    const btn = element('button', 'nav-chat');
    btn.type = 'button';
    btn.title = 'Agent';
    btn.setAttribute('aria-label', 'Agent');
    btn.setAttribute('aria-pressed', String(this.isOpen()));
    btn.classList.toggle('active', this.isOpen());
    const img = element('img', '');
    img.src = '/ui/icons/wand-magic-sparkles.svg';
    img.alt = '';
    const label = element('span', '', 'Agent');
    btn.append(img, label);
    btn.addEventListener('click', () => this.setOpen(!this.isOpen()));
    this.toggle = btn;
    return btn;
  }

  isOpen(): boolean {
    return !this.root.hidden;
  }

  setOpen(open: boolean): void {
    this.root.hidden = !open;
    document.body.classList.toggle('chat-open', open);
    writeOpen(open);
    this.toggle?.setAttribute('aria-pressed', String(open));
    this.toggle?.classList.toggle('active', open);
    if (open && !this.loaded) void this.load();
    if (open) this.input.focus();
  }

  private async load(): Promise<void> {
    this.loaded = true;
    try {
      const session = await agentApi.session();
      this.modelLabel.textContent = `${session.modelId} · ${session.user}`;
      this.modelLabel.title = `${session.provider}: ${session.modelId}, as ${session.user}`;
      this.available = true;
      const conversations = await agentApi.listConversations();
      this.fillConversations(conversations);
      await this.openConversation(conversations[0]?.id ?? null);
    } catch (err) {
      this.available = false;
      this.messages.replaceChildren(
        element('div', 'chat-notice', `The agent service is unavailable: ${(err as Error).message}`)
      );
    }
    this.updateControls();
  }

  private fillConversations(conversations: ConversationInfo[]): void {
    const options = conversations.map((conversation) => {
      const option = element('option', '', conversation.title || 'Untitled conversation');
      option.value = conversation.id;
      return option;
    });
    const fresh = element('option', '', 'New conversation');
    fresh.value = '';
    this.select.replaceChildren(fresh, ...options);
    this.select.value = this.conversationId ?? '';
  }

  private async refreshConversations(): Promise<void> {
    this.fillConversations(await agentApi.listConversations());
  }

  /** Shows conversation `id`, or an empty one that starts with the next message. */
  private async openConversation(id: string | null): Promise<void> {
    if (this.controller) return;
    this.conversationId = id;
    this.select.value = id ?? '';
    this.texts.clear();
    this.toolCalls.clear();
    this.messages.replaceChildren();

    if (id !== null) {
      try {
        const detail = await agentApi.getConversation(id);
        for (const item of detail.items) this.showItem(item);
      } catch (err) {
        this.showError(err as Error);
      }
    }
    this.updateControls();
    this.scrollToEnd(true);
  }

  private async deleteConversation(): Promise<void> {
    const id = this.conversationId;
    if (id === null || this.controller) return;
    try {
      await agentApi.deleteConversation(id);
      this.conversationId = null;
      await this.refreshConversations();
      await this.openConversation(null);
    } catch (err) {
      this.showError(err as Error);
    }
  }

  private showItem(item: TranscriptItem): void {
    switch (item.kind) {
      case 'user':
        this.messages.appendChild(element('div', 'chat-msg chat-user', item.text));
        break;
      case 'assistant': {
        const msg = element('div', 'chat-msg chat-assistant');
        const rendered = element('div', 'chat-rendered');
        rendered.innerHTML = item.html;
        retargetLinks(rendered);
        msg.appendChild(rendered);
        this.messages.appendChild(msg);
        break;
      }
      case 'tool_call': {
        const chip = this.toolChip(item.name);
        chip.classList.add(item.ok === false ? 'failed' : 'done');
        this.messages.appendChild(chip);
        if (item.ok) this.drawRender(chip, item.name, item.args);
        break;
      }
    }
  }

  private async send(): Promise<void> {
    const prompt = this.input.value.trim();
    if (!prompt || this.controller || !this.available) return;

    const controller = new AbortController();
    this.controller = controller;
    this.updateControls();
    this.input.value = '';
    this.messages.appendChild(element('div', 'chat-msg chat-user', prompt));
    this.scrollToEnd(true);

    try {
      if (this.conversationId === null) {
        this.conversationId = (await agentApi.createConversation()).id;
      }
      for await (const event of agentApi.turn(this.conversationId, prompt, controller.signal)) {
        this.handle(event);
      }
    } catch (err) {
      if (controller.signal.aborted) {
        this.messages.appendChild(element('div', 'chat-notice', 'Stopped.'));
      } else {
        this.showError(err as Error);
      }
    } finally {
      this.controller = null;
      this.updateControls();
      try {
        await this.refreshConversations();
      } catch {
        // The list stays as it was.
      }
    }
  }

  private handle(event: AgentEvent): void {
    const nearEnd = this.isNearEnd();

    switch (event.type) {
      case 'TEXT_MESSAGE_CONTENT': {
        const message = this.textMessage(event.messageId);
        message.text += event.delta;
        message.raw.textContent = message.text.slice(message.renderedLength);
        break;
      }
      case 'CUSTOM': {
        const rendered = asRenderedText(event);
        if (rendered === null) break;
        const message = this.textMessage(rendered.messageId);
        message.rendered.innerHTML = rendered.html;
        retargetLinks(message.rendered);
        message.renderedLength = rendered.sourceLength;
        message.raw.textContent = message.text.slice(rendered.sourceLength);
        break;
      }
      case 'TOOL_CALL_START': {
        const chip = this.toolChip(event.toolCallName);
        chip.classList.add('running');
        this.messages.appendChild(chip);
        this.toolCalls.set(event.toolCallId, { chip, name: event.toolCallName, args: '' });
        break;
      }
      case 'TOOL_CALL_ARGS': {
        const call = this.toolCalls.get(event.toolCallId);
        if (call) call.args += event.delta;
        break;
      }
      case 'TOOL_CALL_RESULT': {
        const call = this.toolCalls.get(event.toolCallId);
        if (!call) break;
        call.chip.classList.replace('running', 'done');
        let args: Record<string, unknown> = {};
        try {
          args = call.args ? (JSON.parse(call.args) as Record<string, unknown>) : {};
        } catch {
          // Arguments that do not parse draw nothing.
        }
        this.drawRender(call.chip, call.name, args);
        break;
      }
      case 'RUN_ERROR':
        this.messages.appendChild(element('div', 'chat-error', event.message));
        for (const call of this.toolCalls.values()) call.chip.classList.replace('running', 'failed');
        break;
      default:
        break;
    }

    this.scrollToEnd(nearEnd);
  }

  /** The bubble of text message `messageId`, made when its first text arrives. */
  private textMessage(messageId: string): TextMessage {
    let message = this.texts.get(messageId);
    if (message === undefined) {
      const msg = element('div', 'chat-msg chat-assistant');
      const rendered = element('div', 'chat-rendered');
      const raw = element('div', 'chat-raw');
      msg.append(rendered, raw);
      this.messages.appendChild(msg);
      message = { rendered, raw, text: '', renderedLength: 0 };
      this.texts.set(messageId, message);
    }
    return message;
  }

  private toolChip(name: string): HTMLElement {
    const chip = element('div', 'chat-tool');
    chip.appendChild(element('span', 'chat-tool-name', name));
    return chip;
  }

  /** Draws a render tool call's image beneath its chip, through the webapp's REST route. */
  private drawRender(chip: HTMLElement, name: string, args: Record<string, unknown>): void {
    const render = RENDER_CALLS[name];
    if (render === undefined) return;

    const call = render(args);
    const figure = element('figure', 'chat-figure');
    const caption = element('figcaption', '');
    const link = element('a', '', 'Open in the viewer');
    link.href = call.link;
    caption.appendChild(link);
    figure.appendChild(caption);
    chip.after(figure);

    call
      .image()
      .then((image) => {
        const img = element('img', '');
        img.src = `data:${image.mime_type};base64,${image.png_base64}`;
        img.alt = name;
        figure.prepend(img);
      })
      .catch((err: Error) => {
        figure.prepend(element('div', 'chat-error', `The image could not be drawn: ${err.message}`));
      });
  }

  private showError(err: Error): void {
    const text = err instanceof AgentError ? err.message : `The agent service did not answer: ${err.message}`;
    this.messages.appendChild(element('div', 'chat-error', text));
  }

  private updateControls(): void {
    const running = this.controller !== null;
    this.sendButton.hidden = running;
    this.stopButton.hidden = !running;
    this.sendButton.disabled = !this.available;
    this.input.disabled = !this.available;
    this.select.disabled = running || !this.available;
    this.deleteButton.disabled = running || this.conversationId === null;
  }

  private isNearEnd(): boolean {
    const el = this.messages;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  }

  private scrollToEnd(when: boolean): void {
    if (when) this.messages.scrollTop = this.messages.scrollHeight;
  }

  private resize(handle: HTMLElement, ev: PointerEvent): void {
    ev.preventDefault();
    handle.setPointerCapture(ev.pointerId);
    const rem = REM_PX();
    const move = (e: PointerEvent) => {
      const widthPx = Math.max(rem * 18, Math.min(rem * 60, window.innerWidth - e.clientX));
      document.documentElement.style.setProperty('--chat-w', `${(widthPx / rem).toFixed(3)}rem`);
    };
    const up = (e: PointerEvent) => {
      handle.releasePointerCapture(e.pointerId);
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', up);
      handle.removeEventListener('pointercancel', up);
    };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', up);
    handle.addEventListener('pointercancel', up);
  }
}
