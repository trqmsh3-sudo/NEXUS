import 'dotenv/config';
import React, { useState } from 'react';
import { render, Box, Text, useInput } from 'ink';
import { Agent } from './agent.js';
import { tools } from './tools.js';

const apiKey = process.env.OPENROUTER_API_KEY;

if (!apiKey) {
  // eslint-disable-next-line no-console
  console.error('OPENROUTER_API_KEY is not set. Please add it to your environment or .env file.');
  process.exit(1);
}

const agent = new Agent({
  apiKey,
  model: 'openrouter/auto',
  instructions: 'You are a helpful terminal chat assistant.',
  tools,
  maxSteps: 4
});

const App: React.FC = () => {
  const [input, setInput] = useState('');
  const [history, setHistory] = useState<string[]>([]);

  useInput((inputKey, key) => {
    if (key.return) {
      if (!input.trim()) return;

      const userLine = `You: ${input}`;
      setHistory((h) => [...h, userLine, 'Assistant: ']);

      (async () => {
        let buffer = '';

        agent.removeAllListeners('stream:delta');
        agent.on('stream:delta', (delta) => {
          buffer += delta;
          setHistory((h) => {
            const updated = [...h];
            const last = updated[updated.length - 1] ?? '';
            if (last.startsWith('Assistant:')) {
              updated[updated.length - 1] = `Assistant: ${buffer}`;
            } else {
              updated.push(`Assistant: ${buffer}`);
            }
            return updated;
          });
        });

        await agent.runOnce(input);
      })().catch((err) => {
        // eslint-disable-next-line no-console
        console.error('Agent error:', err);
      });

      setInput('');
    } else if (key.backspace || key.delete) {
      setInput((prev) => prev.slice(0, -1));
    } else if (inputKey) {
      setInput((prev) => prev + inputKey);
    }
  });

  return (
    <Box flexDirection="column">
      <Box flexDirection="column" marginBottom={1}>
        {history.map((line, idx) => (
          <Text key={idx}>{line}</Text>
        ))}
      </Box>
      <Box>
        <Text color="green">{'> '}</Text>
        <Text>{input}</Text>
      </Box>
    </Box>
  );
};

render(<App />);

