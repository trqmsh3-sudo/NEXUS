import 'dotenv/config';
import { Agent } from './agent.js';
import { tools } from './tools.js';

async function main() {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    console.error('OPENROUTER_API_KEY is not set. Please add it to your environment or .env file.');
    process.exit(1);
  }

  const agent = new Agent({
    apiKey,
    model: 'openrouter/auto',
    instructions: 'You are a helpful AI agent.',
    tools,
    maxSteps: 4
  });

  agent.on('stream:delta', (delta) => {
    process.stdout.write(delta);
  });

  const prompt = process.argv.slice(2).join(' ') || 'Hello! Who are you?';
  console.log(`\nUser: ${prompt}\nAssistant: `);

  await agent.runOnce(prompt);
  console.log('\n');
}

main().catch((err) => {
  console.error('Agent error:', err);
  process.exit(1);
});

