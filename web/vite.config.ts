import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(() => ({
    plugins: [react()],
    server: {
        host: "::",
        port: 8080,
        hmr: {
            overlay: false,
        },
    },
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "./src"),
        },
    },
}));
