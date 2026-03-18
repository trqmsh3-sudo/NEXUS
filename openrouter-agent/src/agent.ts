import { OpenRouter, tool, stepCountIs } from '@openrouter/sdk';
import type { Tool, StopCondition, StreamableOutputItem } from '@openrouter/sdk';
import { EventEmitter } from 'eventemitter3';
import { z } from 'zod';

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface AgentEvents {
  'message:user': (message: Message) => void;
  'message:assistant': (message: Message) => void;
  'item:update': (item: StreamableOutputItem) => void;
  'stream:start': () => void;
  'stream:delta': (delta: string, accumulated: string) => void;
  'stream:end': (fullText: string) => void;
  'tool:call': (name: string, args: unknown) => void;
  'tool:result': (name: string, result: unknown) => void;
  'reasoning:update': (text: string) => void;
  'error': (error: Error) => void;
  'thinking:start': () => void;
  'thinking:end': () => void;
}

export interface AgentConfig {
  apiKey: string;
  model?: string;
  instructions?: string;
  tools?: Tool<z.ZodTypeAny, z.ZodTypeAny>[];
  maxSteps?: number;
}

export class Agent extends EventEmitter<AgentEvents> {
  private client: OpenRouter;
  private messages: Message[] = [];
  private config: Required<Omit<AgentConfig, 'apiKey'>> & { apiKey: string };

  constructor(config: AgentConfig) {
    super();
    this.client = new OpenRouter({ apiKey: config.apiKey });
    this.config = {
      apiKey: config.apiKey,
      model: config.model ?? 'openrouter/auto',
      instructions: config.instructions ?? 'You are a helpful assistant.',
      tools: config.tools ?? [],
      maxSteps: config.maxSteps ?? 5
    };
  }

  addUserMessage(content: string) {
    const msg: Message = { role: 'user', content };
    this.messages.push(msg);
    this.emit('message:user', msg);
  }

  getHistory() {
    return [...this.messages];
  }

  async runOnce(prompt: string): Promise<string> {
    this.addUserMessage(prompt);

    try {
      this.emit('stream:start');

      const stop: StopCondition = stepCountIs(this.config.maxSteps);

      const stream = this.client.chat.completions.stream({
        model: this.config.model,
        messages: [
          { role: 'system', content: this.config.instructions },
          ...this.messages
        ],
        tools: this.config.tools,
        stop
      });

      let fullText = '';

      for await (const item of stream) {
        this.emit('item:update', item);

        if (item.type === 'response.reflection_delta') {
          const delta = item.delta;
          if (typeof delta === 'string' && delta.length > 0) {
            this.emit('reasoning:update', delta);
          }
        }

        if (item.type === 'response.output_text_delta') {
          const delta = item.delta;
          if (typeof delta === 'string' && delta.length > 0) {
            fullText += delta;
            this.emit('stream:delta', delta, fullText);
          }
        }

        if (item.type === 'response.completed') {
          const choice = item.response.output[0];
          if (choice && choice.message && typeof choice.message.content === 'string') {
            const assistantMessage: Message = {
              role: 'assistant',
              content: choice.message.content
            };
            this.messages.push(assistantMessage);
            this.emit('message:assistant', assistantMessage);
          }
        }
      }

      this.emit('stream:end', fullText);
      return fullText;
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      this.emit('error', err);
      throw err;
    }
  }
}

export { tool };
