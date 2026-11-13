import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
	await page.route('**/densex/telemetry', (route) => route.fulfill({ json: {}, status: 404 }));
});

test('DenseX sections preserve chat navigation and identify unavailable pages', async ({
	page
}) => {
	await page.goto('/#/console/overview');
	await page.getByLabel('Expand navigation', { exact: true }).click();
	const navigation = page.getByRole('navigation', { name: 'DenseX 控制台' });

	await expect(navigation.getByRole('link')).toHaveCount(7);
	for (const name of ['總覽', '請求與日誌', '硬體', '模型與設定', '文件快取', 'API 連線']) {
		await navigation.getByRole('link', { exact: true, name }).click();
		await expect(page.getByRole('heading', { exact: true, name })).toBeVisible();

		if (name === '總覽' || name === '硬體') {
			await expect(page.getByText('遙測未啟用', { exact: true })).toBeVisible();
		} else {
			await expect(page.getByText('尚未提供', { exact: true })).toBeVisible();
		}

		await expect(navigation.getByRole('link', { exact: true, name })).toHaveAttribute(
			'aria-current',
			'page'
		);
	}
	await navigation.getByRole('link', { exact: true, name: '聊天' }).click();
	await expect(page).toHaveURL(/\/#\/$/);
});

test('mobile section navigation closes the sidebar', async ({ page }) => {
	await page.setViewportSize({ height: 812, width: 375 });
	await page.goto('/#/console/overview');
	await page.getByLabel('Expand navigation', { exact: true }).click();
	await page
		.getByRole('navigation', { name: 'DenseX 控制台' })
		.getByRole('link', { exact: true, name: '硬體' })
		.click();
	await expect(page.getByRole('heading', { exact: true, name: '硬體' })).toBeVisible();
	await expect(page.getByRole('navigation', { name: 'DenseX 控制台' })).toHaveCount(0);
	await expect(page.getByLabel('Expand navigation', { exact: true })).toBeVisible();
});
