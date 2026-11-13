<script lang="ts">
	import { initStores } from '$lib/stores/init';
	import { ApiError, apiFetch } from '$lib/utils/api-fetch';
	import { onMount } from 'svelte';

	const statuses: Record<string, string> = {
		cancelled: '已取消',
		completed: '已完成',
		error: '錯誤',
		processing: '處理中',
		queued: '排隊中'
	};
	const finishes: Record<string, string> = {
		cancelled: '已取消',
		completed: '完成',
		invalid_request: '請求無效',
		length: '長度上限',
		server_error: '伺服器錯誤'
	};
	const reasons: Record<string, string> = {
		'partial task counters unavailable': '部分工作統計未取得',
		'request in progress': '請求仍在處理中',
		'request not tokenized': '請求未經 token 化'
	};
	const eventTypes: Record<string, string> = {
		model_loaded: '模型已載入',
		request_error: '請求發生錯誤'
	};
	const endpoints = new Set([
		'/completion',
		'/completions',
		'/v1/completions',
		'/chat/completions',
		'/v1/chat/completions',
		'/responses',
		'/v1/responses',
		'/audio/transcriptions',
		'/v1/audio/transcriptions',
		'/v1/messages',
		'/infill'
	]);
	type RequestRow = {
		id: number;
		endpoint: string;
		started_at: number | null;
		queue_ms: number | null;
		processing_ms: number | null;
		input_tokens: number | null;
		output_tokens: number | null;
		unavailable_reason: string | null;
		status: string;
		finish_reason: string | null;
		task_count: number | null;
	};
	type EventRow = { sequence: number; at: number | null; code: string };
	type Snapshot = {
		schema_version: 1;
		instance_id: string;
		sampled_at: number | null;
		sequence: number | null;
		scope: 'server_lifetime';
		capacities: {
			requests: number | null;
			events: number | null;
			active_requests: number | null;
			active_tasks: number | null;
		};
		dropped_requests: number | null;
		requests_evicted: number | null;
		events_evicted: number | null;
		requests: RequestRow[];
		events: EventRow[];
	};
	const numeric = (value: unknown): number | null =>
		typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
	const record = (value: unknown): Record<string, unknown> =>
		value && typeof value === 'object' && !Array.isArray(value)
			? (value as Record<string, unknown>)
			: {};
	function sanitize(input: unknown): Snapshot {
		const root = record(input);

		if (
			root.schema_version !== 1 ||
			typeof root.instance_id !== 'string' ||
			!Array.isArray(root.requests) ||
			!Array.isArray(root.events)
		)
			throw new Error('Invalid snapshot');

		const capacities = record(root.capacities);
		const requests: RequestRow[] = [];
		const events: EventRow[] = [];

		for (const entry of root.requests.slice(-256)) {
			const r = record(entry);
			const id = numeric(r.id);

			if (id === null) continue;

			requests.push({
				endpoint:
					typeof r.endpoint === 'string' && endpoints.has(r.endpoint) ? r.endpoint : 'unknown',
				finish_reason:
					typeof r.finish_reason === 'string' && Object.hasOwn(finishes, r.finish_reason)
						? r.finish_reason
						: null,
				id,
				input_tokens: numeric(r.input_tokens),
				output_tokens: numeric(r.output_tokens),
				processing_ms: numeric(r.processing_ms),
				queue_ms: numeric(r.queue_ms),
				started_at: numeric(r.started_at),
				status:
					typeof r.status === 'string' && Object.hasOwn(statuses, r.status) ? r.status : 'unknown',
				task_count: numeric(r.task_count),
				unavailable_reason:
					typeof r.unavailable_reason === 'string' && Object.hasOwn(reasons, r.unavailable_reason)
						? r.unavailable_reason
						: null
			});
		}
		for (const entry of root.events.slice(-128)) {
			const e = record(entry);
			const seq = numeric(e.sequence);

			if (seq === null) continue;

			events.push({
				at: numeric(e.at),
				code: typeof e.code === 'string' && Object.hasOwn(eventTypes, e.code) ? e.code : 'unknown',
				sequence: seq
			});
		}

		return {
			capacities: {
				active_requests: numeric(capacities.active_requests),
				active_tasks: numeric(capacities.active_tasks),
				events: numeric(capacities.events),
				requests: numeric(capacities.requests)
			},
			dropped_requests: numeric(root.dropped_requests),
			events,
			events_evicted: numeric(root.events_evicted),
			instance_id: root.instance_id,
			requests,
			requests_evicted: numeric(root.requests_evicted),
			sampled_at: numeric(root.sampled_at),
			schema_version: 1,
			scope: 'server_lifetime',
			sequence: numeric(root.sequence)
		};
	}
	let snapshot = $state<Snapshot | null>(null);
	let status = $state<'loading' | 'ok' | 'disabled' | 'auth' | 'error'>('loading');
	let filter = $state('all');
	let selected = $state<number | null>(null);
	let view = $state<'requests' | 'events'>('requests');
	const filtered = $derived(
		snapshot?.requests.filter((r) => filter === 'all' || r.status === filter) ?? []
	);
	const detail = $derived(snapshot?.requests.find((r) => r.id === selected));
	const value = (n: number | null | undefined, unit = '') =>
		n == null ? '未取得' : `${n.toLocaleString('zh-TW')}${unit}`;
	const time = (n: number | null) => (n === null ? '未取得' : new Date(n).toLocaleString('zh-TW'));

	onMount(() => {
		let stopped = false;
		let timer: ReturnType<typeof setTimeout>;

		const controller = new AbortController();

		async function poll() {
			try {
				const next = sanitize(
					await apiFetch<unknown>('/densex/requests', {
						authOnly: true,
						cache: 'no-store',
						signal: controller.signal
					})
				);

				if (stopped) return;

				if (snapshot && snapshot.instance_id !== next.instance_id) selected = null;

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
		void initStores()
			.then(() => {
				if (!stopped) void poll();
			})
			.catch(() => {
				if (!stopped) status = 'error';
			});

		return () => {
			stopped = true;
			controller.abort();
			clearTimeout(timer);
		};
	});
	function exportSnapshot() {
		if (!snapshot) return;

		const { instance_id: _instance, ...safe } = snapshot;
		const url = URL.createObjectURL(
			new Blob([JSON.stringify(safe, null, 2)], { type: 'application/json' })
		);
		const link = document.createElement('a');

		link.href = url;
		link.download = 'densex-diagnostics.json';
		link.click();
		URL.revokeObjectURL(url);
	}
</script>

<div class="mt-4">
	{#if status === 'loading'}<p>正在取得請求與事件</p>
	{:else if status === 'disabled'}<p>遙測未啟用</p>

		<p class="mt-2 text-sm text-muted-foreground">
			啟動伺服器時加入 <code>--densex-telemetry</code>
		</p>
	{:else if status === 'auth'}<p role="alert">認證失敗，請檢查 API key</p>
	{:else if status === 'error' && !snapshot}<p role="alert">無法連線至請求與事件端點</p>{/if}

	{#if snapshot && status !== 'disabled' && status !== 'auth'}
		<p class="mb-3 text-sm text-muted-foreground">
			目前記錄生成類請求；認證失敗與其他 API 操作不在清單內。
		</p>

		<div class="flex flex-wrap gap-3 text-sm text-muted-foreground">
			<span>更新時間：{time(snapshot.sampled_at)}</span>{#if status === 'error'}<span role="alert"
					>已過時</span
				>{/if}
		</div>

		<p class="mt-2 text-sm text-muted-foreground">
			伺服器啟動以來：移除請求 {value(snapshot.requests_evicted)} 筆、未記錄請求 {value(
				snapshot.dropped_requests
			)} 筆、移除事件 {value(snapshot.events_evicted)} 筆。
		</p>

		<button
			class="mt-3 rounded border border-border px-3 py-2 text-sm hover:bg-accent focus-visible:outline-2"
			onclick={exportSnapshot}>匯出診斷 JSON</button
		>

		<div aria-label="資料類別" class="mt-5 flex gap-2">
			<button
				aria-pressed={view === 'requests'}
				class="rounded border border-border px-3 py-2 text-sm hover:bg-accent"
				onclick={() => (view = 'requests')}>請求</button
			>

			<button
				aria-pressed={view === 'events'}
				class="rounded border border-border px-3 py-2 text-sm hover:bg-accent"
				onclick={() => (view = 'events')}>事件</button
			>
		</div>

		{#if view === 'requests'}
			<label class="mt-4 block text-sm" for="request-filter">篩選狀態</label>

			<select
				bind:value={filter}
				class="mt-1 rounded border border-border bg-background p-2 text-foreground"
				id="request-filter"
				><option value="all">全部</option
				>{#each Object.entries(statuses) as [key, label] (key)}<option value={key}>{label}</option
					>{/each}</select
			>

			{#if filtered.length === 0}<p class="mt-4 text-muted-foreground">沒有符合的請求</p>{:else}
				<ul class="mt-4 space-y-2">
					{#each filtered as request (request.id)}<li>
							<button
								class="w-full rounded-lg border border-border bg-card p-3 text-left hover:bg-accent focus-visible:outline-2"
								onclick={() => (selected = request.id)}
								><span class="font-medium">請求 #{request.id}</span><span class="ml-3"
									>{statuses[request.status] ?? '未識別'}</span
								><span class="mt-1 block text-sm text-muted-foreground"
									>{time(request.started_at)}</span
								></button
							>
						</li>{/each}
				</ul>
			{/if}

			{#if detail}<section class="mt-5 rounded-lg border border-border bg-card p-4">
					<h2 class="font-semibold">請求 #{detail.id} 詳細資訊</h2>

					<dl class="mt-3 grid gap-2 text-sm sm:grid-cols-2">
						<div>端點：{detail.endpoint === 'unknown' ? '未識別' : detail.endpoint}</div>

						<div>狀態：{statuses[detail.status] ?? '未識別'}</div>

						<div>開始時間：{time(detail.started_at)}</div>

						<div>排隊時間：{value(detail.queue_ms, ' ms')}</div>

						<div>處理時間：{value(detail.processing_ms, ' ms')}</div>

						<div>輸入 token：{value(detail.input_tokens)}</div>

						<div>輸出 token：{value(detail.output_tokens)}</div>

						<div>結束原因：{detail.finish_reason ? finishes[detail.finish_reason] : '未取得'}</div>

						<div>工作數：{value(detail.task_count)}</div>
					</dl>

					{#if detail.unavailable_reason}<p class="mt-2 text-sm text-muted-foreground">
							{reasons[detail.unavailable_reason]}
						</p>{/if}

					<p class="mt-2 text-sm text-muted-foreground">
						排隊時間至首個工作開始；處理時間至最後工作結束，並非各工作時間總和。
					</p>
				</section>{/if}
		{:else}
			<p class="mt-4 text-sm text-muted-foreground">事件僅含模型載入與請求錯誤摘要。</p>

			{#if snapshot.events.length === 0}<p class="mt-4 text-muted-foreground">
					目前沒有事件
				</p>{:else}<ul class="mt-4 space-y-2">
					{#each snapshot.events as event (event.sequence)}<li
							class="rounded-lg border border-border bg-card p-3"
						>
							<span>#{event.sequence} · {eventTypes[event.code] ?? '未識別事件'}</span><span
								class="mt-1 block text-sm text-muted-foreground">{time(event.at)}</span
							>
						</li>{/each}
				</ul>{/if}
		{/if}
	{/if}
</div>
