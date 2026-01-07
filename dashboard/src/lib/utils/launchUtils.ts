export interface LaunchOptions {
    max_input_tokens?: number;
    max_output_tokens?: number;
    temperature?: number;
    kv_cache_bits?: number;
}

export function applyLaunchOptions(instanceData: any, options: LaunchOptions): any {
    if (!instanceData || typeof instanceData !== "object") {
        return instanceData;
    }

    let configTarget = instanceData;
    // Handle TaggedModel serialization (wrapped in { "MlxRingInstance": ... })
    const keys = Object.keys(instanceData);
    if (
        keys.length === 1 &&
        (keys[0] === "MlxRingInstance" ||
            keys[0] === "MlxJacclInstance")
    ) {
        configTarget = instanceData[keys[0]];
    }

    configTarget.config = {
        max_input_tokens: options.max_input_tokens,
        max_output_tokens: options.max_output_tokens,
        temperature: options.temperature,
        kv_cache_bits: options.kv_cache_bits,
    };

    return instanceData;
}
