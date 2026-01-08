import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { appStore } from './app.svelte';

// Mock browser env
vi.stubGlobal('browser', true);

// Mock crypto.randomUUID
const randomUUID = vi.fn(() => 'test-uuid');
vi.stubGlobal('crypto', { randomUUID });

// Mock localStorage
const localStorageMock = (() => {
    let store: Record<string, string> = {};
    return {
        getItem: vi.fn((key: string) => store[key] || null),
        setItem: vi.fn((key: string, value: string) => {
            store[key] = value.toString();
        }),
        clear: vi.fn(() => {
            store = {};
        }),
    };
})();
vi.stubGlobal('localStorage', localStorageMock);

// Mock fetch
const fetchMock = vi.fn();
vi.stubGlobal('fetch', fetchMock);

// Mock AbortController
const abortMock = vi.fn();
const constructorSpy = vi.fn();
class AbortControllerMock {
    signal = { aborted: false };
    abort = abortMock;
    constructor() {
        constructorSpy();
    }
}
vi.stubGlobal('AbortController', AbortControllerMock);

describe('AppStore', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        // Reset store state
        appStore.clearChat();
        appStore.isLoading = false;
        appStore.instances = {
            'node1': {
                'MlxRingInstance': {
                    shardAssignments: {
                        modelId: 'test-model'
                    }
                }
            }
        };
        fetchMock.mockResolvedValue({
            ok: true,
            body: {
                getReader: () => ({
                    read: vi.fn()
                        .mockResolvedValueOnce({ done: false, value: new TextEncoder().encode('data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n') })
                        .mockResolvedValueOnce({ done: true })
                })
            }
        });
    });

    it('should initialize AbortController when sending a message', async () => {
        await appStore.sendMessage('Hello');

        expect(constructorSpy).toHaveBeenCalled();
        expect(fetchMock).toHaveBeenCalledWith(
            expect.stringContaining('/v1/chat/completions'),
            expect.objectContaining({
                signal: expect.anything()
            })
        );
    });

    it('should abort request when stopGeneration is called', async () => {
        // Mock fetch to pending to simulate network delay
        fetchMock.mockReturnValue(new Promise(resolve => setTimeout(resolve, 100)));

        // Start a message generation
        const sendPromise = appStore.sendMessage('Hello');

        // Check immediately (fetch is pending)
        expect(appStore.isLoading).toBe(true);
        expect(appStore.abortController).not.toBeNull();

        // Stop generation
        appStore.stopGeneration();

        expect(abortMock).toHaveBeenCalled();
        expect(appStore.isLoading).toBe(false);
        expect(appStore.abortController).toBeNull();

        // Wait for the "network delay" to cleanup
        await new Promise(resolve => setTimeout(resolve, 150));
        await sendPromise;
    });
});
