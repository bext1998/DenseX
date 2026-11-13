<script lang="ts">
	import { page } from '$app/state';
	import { deviceStore, uiStore } from '$lib/stores';

	const sections = [
		{ label: '聊天', slug: '' },
		{ label: '總覽', slug: 'overview' },
		{ label: '請求與日誌', slug: 'requests' },
		{ label: '硬體', slug: 'hardware' },
		{ label: '模型與設定', slug: 'models' },
		{ label: '文件快取', slug: 'cache' },
		{ label: 'API 連線', slug: 'connections' }
	];
</script>

<nav aria-label="DenseX 控制台" class="flex shrink-0 flex-col gap-1 border-b p-2">
	{#each sections as section (section.slug)}
		{@const active = section.slug
			? page.params.section === section.slug
			: page.route.id?.startsWith('/(chat)')}
		<a
			aria-current={active ? 'page' : undefined}
			class={[
				'rounded-md border px-3 py-2 text-sm font-medium hover:bg-accent focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring',
				active ? 'border-primary bg-accent underline underline-offset-4' : 'border-transparent'
			]}
			href={section.slug ? `#/console/${section.slug}` : '#/'}
			onclick={() => {
				if (deviceStore.isMobile) uiStore.isSidebarExpanded = false;
			}}>{section.label}</a
		>
	{/each}
</nav>
