import { expect, test } from '@playwright/test';

const metric = (value: number | string | null, unit = '', scope = 'gpu_device') => ({
	available: value !== null,
	scope,
	unit,
	value,
	...(value === null ? { reason: '感測器不可用' } : {})
});
const snapshot = {
	hardware: {
		metrics: {
			gpu_power_w: metric(null, 'W'),
			gpu_temperature_c: metric(0, 'C'),
			process_gpu_shared_bytes: metric(1024, 'bytes', 'server_process'),
			process_gpu_shared_delta_bytes: metric(0, 'bytes', 'server_process')
		},
		sample_interval_ms: 1000,
		sampled_at: 1794564000000
	},
	instance_id: 'one',
	metrics: {},
	sample_interval_ms: 1000,
	sampled_at: 1794564000000,
	schema_version: 1,
	slots: []
};

test('disabled and authentication responses are distinct', async ({ page }) => {
	test.setTimeout(60000);
	await page.route('**/densex/telemetry', (route) => route.fulfill({ body: '{}', status: 404 }));
	await page.goto('/#/console/overview');
	await expect(page.getByText('遙測未啟用')).toBeVisible({ timeout: 30000 });
	await expect(page.getByText('--densex-telemetry')).toBeVisible();
	await page.unrouteAll();
	await page.route('**/densex/telemetry', (route) => route.fulfill({ body: '{}', status: 401 }));
	await expect(page.getByText('認證失敗，請檢查 API key')).toBeVisible();
});

test('hardware preserves zero, unavailable and last values on disconnect', async ({ page }) => {
	let calls = 0;

	await page.route('**/densex/telemetry', (route) =>
		++calls === 1 ? route.fulfill({ json: snapshot }) : route.abort()
	);
	await page.goto('/#/console/hardware');
	await expect(page.getByText('0 C', { exact: true })).toBeVisible();
	await expect(page.getByText('感測器不可用')).toBeVisible();
	await expect(page.getByText('已過時')).toBeVisible();
	await expect(page.getByText('0 C', { exact: true })).toBeVisible();
	await expect(page.getByText('較同配置載入基準增加')).toHaveCount(0);
	await expect(page.getByText(/不能單靠 Shared 判斷溢出/)).toBeVisible();
});

test('only the server-provided positive delta shows an increase', async ({ page }) => {
	const increased = structuredClone(snapshot);

	increased.hardware.metrics.process_gpu_shared_delta_bytes = metric(
		512,
		'bytes',
		'server_process'
	);
	await page.route('**/densex/telemetry', (route) => route.fulfill({ json: increased }));
	await page.goto('/#/console/hardware');
	await expect(page.getByText(/較同配置載入基準增加，請配合生成速度確認/)).toBeVisible({
		timeout: 30000
	});
	await expect(page.getByText(/不能單靠 Shared 判斷溢出/)).toBeVisible();
});

test('formats ratios and signed bytes with measurement windows', async ({ page }) => {
	test.setTimeout(60000);
	const data = structuredClone(snapshot);
	const slot = {
		mtp_acceptance_rate: { ...metric(0.5, 'ratio', 'server'), window: 'current_request_average' }
	};
	const response = {
		...data,
		hardware: {
			...data.hardware,
			metrics: {
				...data.hardware.metrics,
				process_gpu_shared_delta_bytes: {
					...metric(-1024, 'bytes', 'server_process'),
					window: 'recent_sample'
				}
			}
		},
		slots: [slot]
	};

	await page.route('**/densex/telemetry', (route) => route.fulfill({ json: response }));
	await page.goto('/#/console/overview');
	await expect(page.getByText('50 %', { exact: true })).toBeVisible({ timeout: 30000 });
	await expect(page.getByText('本次請求平均 · 伺服器')).toBeVisible();
	await page.goto('/#/console/hardware');
	await expect(page.getByText('-1 KiB', { exact: true })).toBeVisible();
	await expect(page.getByText('最近取樣 · 伺服器程序')).toBeVisible();
	await expect(page.getByText(/較同配置載入基準增加/)).toHaveCount(0);
});
