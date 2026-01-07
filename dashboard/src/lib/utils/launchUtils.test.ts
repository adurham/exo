import { describe, it, expect } from 'vitest';
import { applyLaunchOptions } from './launchUtils';

describe('applyLaunchOptions', () => {
    it('applies options to flat instance object', () => {
        const instance = { id: 'test', other: 'field' };
        const options = { temperature: 0.7, kv_cache_bits: 4 };

        applyLaunchOptions(instance, options);

        expect((instance as any).config).toEqual({
            max_input_tokens: undefined,
            max_output_tokens: undefined,
            temperature: 0.7,
            kv_cache_bits: 4
        });
    });

    it('applies options to nested TaggedModel (MlxRingInstance)', () => {
        const inner = { id: 'test' };
        const instance = { MlxRingInstance: inner };
        const options = { max_output_tokens: 100 };

        applyLaunchOptions(instance, options);

        expect((inner as any).config).toBeDefined();
        expect((inner as any).config.max_output_tokens).toBe(100);
        // Ensure structure is preserved
        expect(instance.MlxRingInstance).toBe(inner);
    });

    it('applies options to nested TaggedModel (MlxJacclInstance)', () => {
        const inner = { id: 'test' };
        const instance = { MlxJacclInstance: inner };
        const options = { temperature: 0.1 };

        applyLaunchOptions(instance, options);

        expect((inner as any).config.temperature).toBe(0.1);
    });
});
