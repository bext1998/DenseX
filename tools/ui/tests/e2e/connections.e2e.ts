import { expect, test } from '@playwright/test';

test('anonymous connection shows safe examples and sends only one manual test', async ({
	context,
	page
}) => {
	test.setTimeout(60000);
	await context.grantPermissions(['clipboard-read', 'clipboard-write']);
	let posts = 0;
	let props = 0;

	await page.route('**/props**', (route) => {
		props++;

		return route.fulfill({ json: {} });
	});

	await page.route('**/v1/models', (route) =>
		route.fulfill({ json: { data: [{ id: "model'quoted" }] } })
	);
	await page.route('**/v1/chat/completions', async (route) => {
		posts++;
		expect(route.request().postDataJSON()).toMatchObject({
			chat_template_kwargs: { enable_thinking: false },
			max_tokens: 8,
			model: "model'quoted",
			stream: false
		});
		await new Promise((resolve) => setTimeout(resolve, 500));
		await route.fulfill({ json: { choices: [] } });
	});
	page.on('pageerror', (error) => console.error('頁面錯誤:', error.message));
	const navigation = await page.goto('/#/console/connections');

	expect(navigation?.status(), await navigation?.text()).toBe(200);
	await expect(page.getByText('不需要 API key')).toBeVisible({ timeout: 50000 });
	expect(props).toBe(0);
	expect(posts).toBe(0);
	await page.getByRole('button', { name: '複製 curl 範例' }).click();
	const copied = await page.evaluate(() => navigator.clipboard.readText());

	expect(copied).toContain('model');
	expect(copied).not.toContain('Authorization: Bearer');
	await page.getByRole('button', { name: '送出測試請求' }).click();
	await expect(page.getByRole('button', { name: '測試中' })).toBeDisabled();
	await expect(page.getByText(/測試成功/)).toBeVisible();
	expect(posts).toBe(1);
});

test('authentication required and unsafe model paths never appear', async ({ page }) => {
	test.setTimeout(60000);
	let requests = 0;

	await page.route('**/v1/models', (route) => {
		requests++;

		return requests === 1
			? route.fulfill({ body: 'secret C:\\private\\model.gguf', status: 401 })
			: route.fulfill({ json: { data: [{ id: 'C:\\private\\model.gguf' }] } });
	});
	await page.goto('/#/console/connections');
	await expect(page.getByText('需要 API key', { exact: true })).toBeVisible({ timeout: 30000 });
	await expect(page.getByText(/安全 alias/)).toBeVisible();
	await expect(page.locator('body')).not.toContainText('C:\\private\\model.gguf');
	await expect(page.locator('body')).not.toContainText('secret');
	await expect(page.getByRole('button', { name: '送出測試請求' })).toBeDisabled();
});

test('configured key stays private and failure is generic', async ({ context, page }) => {
	test.setTimeout(60000);
	await context.grantPermissions(['clipboard-read', 'clipboard-write']);
	await page.addInitScript(() =>
		localStorage.setItem('LlamaUi.config', JSON.stringify({ apiKey: 'test-secret-42' }))
	);
	let probes = 0;

	await page.route('**/v1/models', (route) => {
		probes++;

		if (probes === 1) {
			expect(route.request().headers().authorization).toBeUndefined();

			return route.fulfill({ body: '{}', status: 401 });
		}

		expect(route.request().headers().authorization).toBe('Bearer test-secret-42');

		return route.fulfill({ json: { data: [{ id: 'safe-model' }] } });
	});
	await page.route('**/v1/chat/completions', (route) =>
		route.fulfill({ body: 'test-secret-42 C:\\private\\model', status: 500 })
	);
	await page.goto('/#/console/connections');
	await expect(page.getByText('需要 API key', { exact: true })).toBeVisible({ timeout: 30000 });
	await page.getByRole('button', { name: '複製 curl 範例' }).click();
	const copied = await page.evaluate(() => navigator.clipboard.readText());

	expect(copied).toContain('$DENSEX_API_KEY');
	expect(copied).not.toContain('test-secret-42');
	await page.getByRole('button', { name: '送出測試請求' }).click();
	await expect(page.getByText(/測試失敗/)).toBeVisible();
	await expect(page.locator('body')).not.toContainText('test-secret-42');
	await expect(page.locator('body')).not.toContainText('C:\\private\\model');
});
