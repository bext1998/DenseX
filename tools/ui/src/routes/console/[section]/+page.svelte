<script lang="ts">
	import { page } from '$app/state';
	import { ApiError, apiFetch } from '$lib/utils/api-fetch';

	type Metric = {
		available: boolean;
		value: number | string | null;
		scope: string;
		unit: string;
		reason?: string;
		source?: string;
		window?: string;
	};
	type Snapshot = {
		schema_version: number;
		instance_id: string;
		sampled_at: number;
		server_sampled_at?: number;
		metrics: Record<string, Metric>;
		slots: Record<string, Metric>[];
		last_request: Record<string, Metric> | null;
		hardware: { sampled_at: string | null; metrics: Record<string, Metric> };
	};
	const titles: Record<string, string> = {
		cache: '文件快取',
		connections: 'API 連線',
		hardware: '硬體',
		models: '模型與設定',
		overview: '總覽',
		requests: '請求與日誌'
	};
	const scopes: Record<string, string> = {
		current_request: '本次請求',
		current_response: '本次回答',
		deferred_queue: '等待佇列',
		gpu_device: 'GPU 裝置',
		last_request: '上次請求',
		last_response: '上次回答',
		'NVML device index 0, entire GPU': 'GPU 0 整體',
		post_load_idle_baseline: '載入後閒置基準',
		recent_sample: '最近取樣',
		server: '伺服器',
		server_process: '伺服器程序',
		'server process, all adapters': '伺服器程序，所有顯示卡',
		'server process working set': '伺服器程序工作集',
		since_post_load_idle_baseline: '較載入後閒置基準',
		since_startup: '啟動以來',
		system: '整台系統',
		'system physical memory': '系統實體記憶體'
	};
	const windows: Record<string, string> = {
		current_request: '本次請求快照',
		current_request_average: '本次請求平均',
		last_request_average: '上次請求平均',
		last_request_completed: '上次請求結束值',
		latest_snapshot: '目前快照',
		post_load_idle_baseline: '載入後首次有效閒置基準',
		recent_sample: '最近取樣',
		since_post_load_idle_baseline: '較該基準'
	};
	const overviewFields = [
		['state', '狀態'],
		['generation_phase', '生成階段'],
		['active_slots', '使用中工作槽'],
		['queued_requests', '排隊請求']
	];
	const slotFields = [
		['state', '狀態'],
		['context_used_tokens', '上下文已用'],
		['context_capacity_tokens', '上下文容量'],
		['prompt_processed_tokens', '讀入 token'],
		['prompt_cached_tokens', '快取 token'],
		['generated_tokens', '生成 token'],
		['prompt_tokens_per_second', '讀入速度'],
		['generation_tokens_per_second', '生成速度'],
		['draft_tokens', '草稿 token'],
		['draft_accepted_tokens', '接受草稿 token'],
		['mtp_acceptance_rate', 'MTP 接受率']
	];
	const hardwareFields = [
		['gpu_name', 'GPU'],
		['process_ram_bytes', '伺服器 RAM'],
		['system_ram_used_bytes', '系統 RAM 已用'],
		['system_ram_total_bytes', '系統 RAM 總量'],
		['gpu_vram_used_bytes', 'VRAM 已用'],
		['gpu_vram_total_bytes', 'VRAM 總量'],
		['gpu_temperature_c', 'GPU 溫度'],
		['gpu_graphics_clock_mhz', 'GPU 核心時脈'],
		['gpu_memory_clock_mhz', 'GPU 記憶體時脈'],
		['gpu_power_w', 'GPU 功率'],
		['process_gpu_dedicated_bytes', '程序 Dedicated GPU 記憶體'],
		['process_gpu_shared_bytes', '程序 Shared GPU 記憶體'],
		['process_gpu_shared_baseline_bytes', 'Shared 載入基準'],
		['process_gpu_shared_delta_bytes', 'Shared 較基準變化']
	];
	let snapshot = $state<Snapshot | null>(null);
	let status = $state<'loading' | 'ok' | 'disabled' | 'auth' | 'error'>('loading');
	const section = $derived(page.params.section ?? '');
	const title = $derived(titles[section] ?? '控制台');
	const monitored = $derived(section === 'overview' || section === 'hardware');

	$effect(() => {
		if (!monitored) return;

		let stopped = false;
		let timer: ReturnType<typeof setTimeout>;

		const controller = new AbortController();

		async function poll() {
			try {
				const next = await apiFetch<Snapshot>('/densex/telemetry', {
					authOnly: true,
					signal: controller.signal
				});

				if (stopped) return;

				if (next.schema_version !== 1) throw new Error('Unsupported telemetry schema');

				snapshot = next;
				status = 'ok';
			} catch (error) {
				if (stopped) return;

				status =
					error instanceof ApiError && error.status === 404
						? 'disabled'
						: error instanceof ApiError && error.status === 401
							? 'auth'
							: 'error';
			}

			if (!stopped) timer = setTimeout(poll, 1000);
		}
		void poll();

		return () => {
			stopped = true;
			controller.abort();
			clearTimeout(timer);
		};
	});
	function display(metric: Metric | undefined) {
		if (!metric?.available || metric.value === null) return '未取得';

		if (typeof metric.value === 'number' && metric.unit === 'ratio') {
			return `${(metric.value * 100).toLocaleString('zh-TW', { maximumFractionDigits: 2 })} %`;
		}

		if (typeof metric.value === 'number' && metric.unit === 'bytes') {
			const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];

			let value = Math.abs(metric.value);
			let index = 0;

			while (value >= 1024 && index < units.length - 1) {
				value /= 1024;
				index++;
			}

			return `${metric.value < 0 ? '-' : ''}${value.toLocaleString('zh-TW', { maximumFractionDigits: 2 })} ${units[index]}`;
		}

		if (metric.unit === 'state' && typeof metric.value === 'string') {
			const states: Record<string, string> = {
				generating: '生成中（思考／回答未區分）',
				idle: '閒置',
				prompt: '讀入中',
				queued: '排隊中'
			};

			return states[metric.value] ?? metric.value;
		}

		return `${typeof metric.value === 'number' ? metric.value.toLocaleString('zh-TW', { maximumFractionDigits: 2 }) : metric.value}${metric.unit ? ` ${metric.unit}` : ''}`;
	}
	function time(value: string | number | null | undefined) {
		return value == null ? '未取得' : new Date(value).toLocaleString('zh-TW');
	}
	function reason(text: string) {
		const labels: Record<string, string> = {
			'API unavailable or call failed': '感測來源尚未就緒或無法讀取',
			'baseline or current sample unavailable': '尚無可比較的基準或目前取樣',
			'counter instances unavailable': '尚未取得程序的 GPU 記憶體取樣',
			'counter read failed': '無法讀取 GPU 記憶體取樣',
			'counter unavailable': '此記憶體感測來源無法使用',
			'generation timing unavailable': '尚無生成計時',
			'invalid process counter instance': '程序的 GPU 記憶體取樣無效',
			'MTP draft tokens unavailable': '尚無可用的 MTP 草稿統計',
			'no process counter instances': '目前沒有此程序的 GPU 記憶體取樣',
			'not sampled yet': '等待首次取樣',
			'no valid idle baseline yet': '等待載入完成後的閒置基準',
			'prompt timing unavailable': '尚無讀入計時',
			'reasoning and final phase are not available in slot state': '此版本無法區分思考與最終回答',
			'unsupported platform': '此平台尚未提供這項感測'
		};

		return labels[text] ?? text;
	}
</script>

{#snippet cards(fields: string[][], metrics: Record<string, Metric>)}
	<div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
		{#each fields as [key, label] (key)}
			{@const metric = metrics[key]}
			<div
				class="min-w-0 rounded-lg border border-border bg-card p-4 text-card-foreground"
				title={metric?.source}
			>
				<div class="text-sm text-muted-foreground">{label}</div>

				<div class="mt-1 break-words text-lg font-semibold">{display(metric)}</div>

				{#if metric?.scope || metric?.window}<div class="mt-1 text-xs text-muted-foreground">
						{[
							metric.window && (windows[metric.window] ?? metric.window),
							metric.scope && (scopes[metric.scope] ?? metric.scope)
						]
							.filter(Boolean)
							.join(' · ')}
					</div>{/if}

				{#if !metric?.available && metric?.reason}<div
						class="mt-1 break-words text-sm text-muted-foreground"
					>
						{reason(metric.reason)}
					</div>{/if}
			</div>
		{/each}
	</div>
{/snippet}

<svelte:head><title>{title} · DenseX</title></svelte:head>
<main class="mx-auto max-w-4xl px-6 pt-24 pb-12 md:pt-12">
	<h1 class="text-2xl font-semibold">{title}</h1>

	{#if !monitored}<p class="mt-4 text-muted-foreground">尚未提供</p>
	{:else}
		{#if status === 'disabled'}<div class="mt-4">
				<p>遙測未啟用</p>

				<p class="mt-2 text-sm text-muted-foreground">
					啟動伺服器時加入 <code>--densex-telemetry</code>
				</p>
			</div>
		{:else if status === 'auth'}<p class="mt-4" role="alert">認證失敗，請檢查 API key</p>
		{:else if status === 'loading'}<p class="mt-4 text-muted-foreground">正在取得遙測</p>
		{:else if status === 'error' && !snapshot}<p class="mt-4" role="alert">
				無法連線至遙測端點
			</p>{/if}

		{#if snapshot && status !== 'disabled' && status !== 'auth'}
			<div class="mt-4 flex flex-wrap gap-x-4 gap-y-1 text-sm text-muted-foreground">
				<span
					>更新時間：{time(
						section === 'hardware'
							? snapshot.hardware.sampled_at
							: (snapshot.server_sampled_at ?? snapshot.sampled_at)
					)}</span
				>{#if status === 'error'}<span role="alert">已過時</span>{/if}
			</div>

			{#if section === 'overview'}
				<div class="mt-4">{@render cards(overviewFields, snapshot.metrics)}</div>

				{#each snapshot.slots as slot, index (slot.slot_id?.value ?? index)}<section class="mt-6">
						<h2 class="mb-3 text-lg font-semibold">
							工作槽 {slot.slot_id?.available ? slot.slot_id.value : index}
						</h2>

						{@render cards(slotFields, slot)}
					</section>{/each}

				{#if snapshot.slots.length === 0 && snapshot.last_request}
					<section class="mt-6">
						<h2 class="mb-3 text-lg font-semibold">上次請求</h2>

						{@render cards(slotFields, snapshot.last_request)}
					</section>
				{/if}
			{:else}
				<div class="mt-4">{@render cards(hardwareFields, snapshot.hardware.metrics)}</div>

				{#if snapshot.hardware.metrics.process_gpu_shared_delta_bytes?.available && Number(snapshot.hardware.metrics.process_gpu_shared_delta_bytes.value) > 0}<p
						class="mt-4"
					>
						較同配置載入基準增加，請配合生成速度確認。
					</p>{/if}

				<p class="mt-3 text-sm text-muted-foreground">不能單靠 Shared 判斷溢出。</p>
			{/if}
		{/if}
	{/if}
</main>
