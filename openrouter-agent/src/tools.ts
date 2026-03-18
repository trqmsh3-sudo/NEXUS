import { tool } from '@openrouter/sdk';
import { z } from 'zod';

export const echoTool = tool({
  name: 'echo',
  description: 'Echo back a message, useful for debugging.',
  inputSchema: z.object({
    text: z.string().describe('The text to echo back')
  }),
  outputSchema: z.object({
    text: z.string().describe('The echoed text')
  }),
  async execute(input) {
    return { text: input.text };
  }
});

export const tools = [echoTool];
