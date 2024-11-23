import { expect, beforeAll, afterAll, afterEach } from "bun:test";

// Extend global test environment
declare global {
  var testSetup: boolean;
}

beforeAll(() => {
  // Initialize any global test setup here
  global.testSetup = true;
  console.log("🚀 Test environment initialized");
});

afterAll(() => {
  // Cleanup after all tests
  global.testSetup = false;
  console.log("✨ Test environment cleaned up");
});

afterEach(() => {
  // Cleanup after each test
  // Add any cleanup logic needed between tests
});

// Export commonly used test utilities
export { expect };