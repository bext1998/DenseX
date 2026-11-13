<script lang="ts">
	import { initStores } from '$lib/stores/init';
	import { settingsStore } from '$lib/stores/settings/index.svelte';
	import { apiFetch } from '$lib/utils/api-fetch';
	import { onMount } from 'svelte';

	type Models = { data: { id: string }[] };
	let auth = $state<'loading' | 'required' | 'none' | 'unknown'>('loading');
	let models = $state<string[]>([]);
	let selected = $state('');
	let unsafe = $state(false);
	let baseUrl = $state('');
	let busy = $state(false);
	let result = $state('');
	let copied = $state('');
	let controller: AbortController;
	const valid = $derived(auth !== 'loading' && auth !== 'unknown' && selected !== '');
	const quoteShell = (value: string) => "'" + value.replaceAll("'", "'\"'\"'") + "'";
	const keyHelp = $derived(
		auth === 'required'
			? '在客戶端設定 API key，範例從環境變數 DENSEX_API_KEY 讀取。'
			: '客戶端可不提供 API key；若必填可使用 not-required。'
	);
	const curl = $derived(
		`curl ${quoteShell(baseUrl + '/chat/completions')} \\\n  -H 'Content-Type: application/json'${auth === 'required' ? ' \\\n  -H "Authorization: Bearer $DENSEX_API_KEY"' : ''} \\\n  -d ${quoteShell(JSON.stringify({ chat_template_kwargs: { enable_thinking: false }, max_tokens: 8, messages: [{ content: 'Hello', role: 'user' }], model: selected, stream: false }))}`
	);
	const python = $derived(
		`import os\nfrom openai import OpenAI\n\nclient = OpenAI(base_url=${JSON.stringify(baseUrl)}, api_key=${auth === 'required' ? 'os.environ["DENSEX_API_KEY"]' : '"not-required"'})\nresponse = client.chat.completions.create(\n    model=${JSON.stringify(selected)},\n    messages=[{"role": "user", "content": "Hello"}],\n    max_tokens=8,\n    extra_body={"chat_template_kwargs": {"enable_thinking": False}},\n)\nprint(response.choices[0].message.content)`
	);

	onMount(() => {
		controller = new AbortController();
		baseUrl = `${location.origin}/v1`;
		async function load() {
			try {
				await initStores();

				if (controller.signal.aborted) return;

				const response = await fetch('/v1/models', {
					cache: 'no-store',
					credentials: 'omit',
					signal: controller.signal
				});

				let data: Models;

				if (response.status === 200) {
					auth = 'none';
					data = await response.json();
				} else if (response.status === 401) {
					auth = 'required';
					data = await apiFetch<Models>('/v1/models', {
						authOnly: true,
						cache: 'no-store',
						signal: controller.signal
					});
				} else {
					auth = 'unknown';

					return;
				}

				if (controller.signal.aborted) return;

				const key = settingsStore.config.apiKey?.toString().trim();

				for (const item of data.data) {
					if (typeof item.id !== 'string') continue;

					if (/^(?:[a-zA-Z]:[\\/]|[\\/])/.test(item.id) || (key && item.id.includes(key))) {
						unsafe = true;
					} else {
						models.push(item.id);
					}
				}
				selected = models[0] ?? '';
			} catch {
				if (!controller.signal.aborted && auth !== 'required') auth = 'unknown';
			}
		}
		void load();

		return () => controller.abort();
	});

	async function copy(label: string, value: string) {
		try {
			await navigator.clipboard.writeText(value);
			copied = `${label}已複製`;
		} catch {
			copied = `${label}複製失敗`;
		}
	}
	async function testConnection() {
		if (busy || !valid) return;

		busy = true;
		result = '';
		const start = performance.now();

		try {
			const body = JSON.stringify({
				chat_template_kwargs: { enable_thinking: false },
				max_tokens: 8,
				messages: [{ content: 'Hello', role: 'user' }],
				model: selected,
				stream: false
			});

			if (auth === 'required') {
				await apiFetch('/v1/chat/completions', { body, method: 'POST', signal: controller.signal });
			} else {
				const response = await fetch('/v1/chat/completions', {
					body,
					credentials: 'omit',
					headers: { 'Content-Type': 'application/json' },
					method: 'POST',
					signal: controller.signal
				});

				if (!response.ok) throw new Error('request failed');
			}

			if (!controller.signal.aborted)
				result = `測試成功，耗時 ${((performance.now() - start) / 1000).toFixed(2)} 秒`;
		} catch {
			if (!controller.signal.aborted)
				result = `測試失敗，耗時 ${((performance.now() - start) / 1000).toFixed(2)} 秒；請檢查連線與 API key。`;
		} finally {
			busy = false;
		}
	}
</script>

<div class="mt-4 space-y-6">
	<section class="rounded-lg border border-border bg-card p-4 text-card-foreground">
		<h2 class="font-semibold">Base URL</h2>

		<p class="mt-2 break-all">{baseUrl}</p>

		<button
			class="mt-3 rounded border border-border px-3 py-2 text-sm hover:bg-accent focus-visible:outline-2"
			onclick={() => copy('Base URL', baseUrl)}>複製 Base URL</button
		>
	</section>

	<section class="rounded-lg border border-border bg-card p-4 text-card-foreground">
		<h2 class="font-semibold">認證與模型</h2>

		{#if auth === 'loading'}<p class="mt-2">正在檢查連線</p>
		{:else if auth === 'unknown'}<p class="mt-2">無法確認 API key 需求，請檢查伺服器連線。</p>
		{:else}
			<p class="mt-2">{auth === 'required' ? '需要 API key' : '不需要 API key'}</p>

			<p class="mt-1 text-sm text-muted-foreground">
				{keyHelp}
			</p>

			<button
				class="mt-3 rounded border border-border px-3 py-2 text-sm hover:bg-accent focus-visible:outline-2"
				onclick={() => copy('API key 設定說明', keyHelp)}>複製 API key 設定說明</button
			>

			{#if unsafe}<p class="mt-3">模型 ID 包含不適合顯示的資訊，請設定安全 alias 後再使用。</p>{/if}

			{#if models.length}
				<label class="mt-4 block text-sm" for="connection-model">模型 ID</label>

				<select
					bind:value={selected}
					class="mt-1 w-full rounded border border-border bg-background p-2 text-foreground"
					id="connection-model"
					>{#each models as model (model)}<option value={model}>{model}</option>{/each}</select
				>

				<button
					class="mt-3 rounded border border-border px-3 py-2 text-sm hover:bg-accent focus-visible:outline-2"
					onclick={() => copy('模型 ID', selected)}>複製模型 ID</button
				>
			{:else if !unsafe}<p class="mt-3">
					{auth === 'required'
						? '無法取得模型 ID，請先在左側設定填入此伺服器的 API key，再重新整理本頁。'
						: '無法取得模型 ID，請檢查伺服器狀態。'}
				</p>{/if}
		{/if}
	</section>

	<section>
		<h2 class="text-lg font-semibold">請求範例（文件範例）</h2>

		{#if valid}
			<h3 class="mt-4 font-medium">curl（POSIX shell）</h3>

			<pre
				class="mt-2 overflow-x-auto rounded-lg border border-border bg-card p-4 text-sm"
				data-testid="connection-curl"><code>{curl}</code></pre>

			<button
				class="mt-2 rounded border border-border px-3 py-2 text-sm hover:bg-accent focus-visible:outline-2"
				onclick={() => copy('curl 範例', curl)}>複製 curl 範例</button
			>

			<h3 class="mt-4 font-medium">Python（openai 套件）</h3>

			<pre
				class="mt-2 overflow-x-auto rounded-lg border border-border bg-card p-4 text-sm"
				data-testid="connection-python"><code>{python}</code></pre>

			<button
				class="mt-2 rounded border border-border px-3 py-2 text-sm hover:bg-accent focus-visible:outline-2"
				onclick={() => copy('Python 範例', python)}>複製 Python 範例</button
			>
		{:else}<p class="mt-2 text-sm text-muted-foreground">取得安全的模型 ID 後提供範例。</p>{/if}
	</section>

	<p aria-live="polite" class="text-sm">{copied}</p>

	<section>
		<button
			class="rounded border border-border bg-primary px-4 py-2 text-primary-foreground disabled:opacity-50 focus-visible:outline-2"
			disabled={!valid || busy}
			onclick={testConnection}>{busy ? '測試中' : '送出測試請求'}</button
		>

		<p aria-live="polite" class="mt-2 text-sm">{result}</p>
	</section>
</div>
