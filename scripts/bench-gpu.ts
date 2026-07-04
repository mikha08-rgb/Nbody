/**
 * Headless driver for the browser GPU benchmark (bench.html): serves the
 * app with Vite, opens the page in full Chromium (new-headless — the
 * headless shell has no WebGPU), and relays the page's report to stdout.
 * README numbers come from re-running this, never from editing.
 *
 * Run with `npm run bench:gpu`. Requires a WebGPU-capable GPU; the page
 * says so and exits cleanly if there is none.
 */
import { chromium } from 'playwright';
import { createServer } from 'vite';

const server = await createServer({ logLevel: 'silent', server: { port: 0 } });
await server.listen();
const base = server.resolvedUrls?.local[0];
if (base === undefined) throw new Error('Vite dev server reported no local URL');

const browser = await chromium.launch({ headless: true, channel: 'chromium' });
const page = await browser.newPage();
page.on('console', (message) => {
  const text = message.text();
  if (text.startsWith('[bench] ')) {
    console.log(text.slice('[bench] '.length));
  } else if (message.type() === 'error' || message.type() === 'warning') {
    // WebGPU validation problems arrive as page console errors, not
    // exceptions — swallowing them would let a broken kernel publish a
    // clean-looking table of garbage numbers.
    console.error(`[page ${message.type()}] ${text}`);
  }
});
page.on('pageerror', (error) => console.error(`page error: ${error.message}`));

await page.goto(`${base}bench.html?auto`);
// The 500k sizes can legitimately take minutes; the skip rule bounds the
// total, but leave generous headroom before declaring it wedged.
await page.waitForFunction(() => document.body.dataset.benchDone === '1', undefined, {
  timeout: 30 * 60_000,
});

await browser.close();
await server.close();
