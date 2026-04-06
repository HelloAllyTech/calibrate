import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    globals: true,
    include: ["tests/**/*.test.tsx", "tests/**/*.test.ts"],
    testTimeout: 15000,
  },
  esbuild: {
    jsx: "automatic",
  },
});
