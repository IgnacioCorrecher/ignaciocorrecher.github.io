import { defineConfig } from 'astro/config';

export default defineConfig({
    site: 'https://ignaciocorrecher.github.io',
    base: '/',
    vite: {
        envPrefix: 'GITHUB_',
    },
});
