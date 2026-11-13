import { expect, test } from '@playwright/test';

const row = (id: number, status = 'completed') => ({
	endpoint: '/v1/chat/completions',
	finish_reason: 'completed',
	id,
	input_tokens: 10,
	output_tokens: 2,
	path: 'C:\\private\\model',
	processing_ms: 100,
	prompt: 'fakeclientsecret',
	queue_ms: 0,
	started_at: 1794564000000,
	status,
	task_count: 1,
	unavailable_reason: null
});
const snapshot = {
	capacities: { active_requests: 128, events: 128, requests: 128 },
	dropped_requests: 1,
	events: [
		{ at: 1794564000000, code: 'model_loaded', content: 'fakeclientsecret', sequence: 1 },
		{ at: 1794564000000, code: 'fakeclientsecret C:\\private\\model', sequence: 2 }
	],
	events_evicted: 2,
	instance_id: 'instance-a',
	requests: [
		row(1),
		row(2, 'processing'),
		{ ...row(3), endpoint: 'C:\\private\\model', status: 'fakeclientsecret' }
	],
	requests_evicted: 3,
	sampled_at: 1794564000000,
	schema_version: 1,
	scope: 'server_lifetime',
	secret: 'fakeclientsecret',
	sequence: 2
};

test('filter, details and export use safe unfiltered snapshot', async ({ page }) => {
	await page.route('**/densex/requests', (route) => route.fulfill({ json: snapshot }));
	await page.goto('/#/console/requests');
	await expect(page.getByRole('button', { name: /請求 #1/ })).toBeVisible();
	await page.getByLabel('篩選狀態').selectOption('processing');
	await expect(page.getByRole('button', { name: /請求 #1/ })).toHaveCount(0);
	await page.getByRole('button', { name: /請求 #2/ }).click();
	await expect(page.getByText('排隊時間：0 ms')).toBeVisible();
	const downloadPromise = page.waitForEvent('download');

	await page.getByRole('button', { name: '匯出診斷 JSON' }).click();
	const download = await downloadPromise;
	const stream = await download.createReadStream();

	let content = '';

	for await (const chunk of stream!) content += chunk.toString();
	const exported = JSON.parse(content);

	expect(exported.requests).toHaveLength(3);
	expect(content).not.toContain('fakeclientsecret');
	expect(content).not.toContain('private');
	await expect(page.locator('body')).not.toContainText('fakeclientsecret');
	await page.getByRole('button', { exact: true, name: '事件' }).click();
	await expect(page.getByText(/模型已載入/)).toBeVisible();
});

test('disabled and authentication are distinct', async ({ page }) => {
	await page.route('**/densex/requests', (route) => route.fulfill({ json: {}, status: 404 }));
	await page.goto('/#/console/requests');
	await expect(page.getByText('遙測未啟用')).toBeVisible();
	await page.unrouteAll();
	await page.route('**/densex/requests', (route) => route.fulfill({ json: {}, status: 401 }));
	await expect(page.getByText('認證失敗，請檢查 API key')).toBeVisible();
});

test('stale values remain and new instance replaces them', async ({ page }) => {
	let calls = 0;
	let replaceInstance = false;

	await page.route('**/densex/requests', (route) => {
		calls++;

		if (calls === 1) return route.fulfill({ json: snapshot });

		if (!replaceInstance) return route.abort();

		return route.fulfill({ json: { ...snapshot, instance_id: 'instance-b', requests: [row(7)] } });
	});
	await page.goto('/#/console/requests');
	await expect(page.getByText('已過時')).toBeVisible();
	await expect(page.getByRole('button', { name: /請求 #1/ })).toBeVisible();
	await page.getByRole('button', { name: /請求 #1/ }).click();
	replaceInstance = true;
	await expect(page.getByRole('button', { name: /請求 #7/ })).toBeVisible();
	await expect(page.getByRole('button', { name: /請求 #1/ })).toHaveCount(0);
	await expect(page.getByText('請求 #1 詳細資訊')).toHaveCount(0);
});

test('display is bounded even if the response is oversized', async ({ page }) => {
	await page.route('**/densex/requests', (route) =>
		route.fulfill({
			json: { ...snapshot, requests: Array.from({ length: 300 }, (_, id) => row(id)) }
		})
	);
	await page.goto('/#/console/requests');
	await expect(page.getByRole('button', { name: /請求 #/ })).toHaveCount(256);
});
