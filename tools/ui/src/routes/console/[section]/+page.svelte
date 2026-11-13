<script lang="ts">
	import { page } from '$app/state';

	const titles: Record<string, string> = {
		cache: '文件快取',
		connections: 'API 連線',
		hardware: '硬體',
		models: '模型與設定',
		overview: '總覽',
		requests: '請求與日誌'
	};
	const title = $derived(titles[page.params.section ?? ''] ?? '控制台');
</script>

<svelte:head><title>{title} · DenseX</title></svelte:head>

<main class="mx-auto max-w-4xl px-6 pt-24 pb-12 md:pt-12">
	<h1 class="text-2xl font-semibold">{title}</h1>

	<p class="mt-4 text-muted-foreground">尚未提供</p>
</main>
