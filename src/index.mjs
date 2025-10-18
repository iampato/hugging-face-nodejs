import { pipeline } from '@huggingface/transformers';

// helpers
function extractEmbedding(result) {
    if (!result || result.length === 0) {
        throw new Error("No tensor result provided.");
    }

    const tensor = result[0];

    // Try both common locations where data may reside
    const floatArray =
        tensor.data ||
        tensor.cpuData ||
        tensor.ort_tensor?.cpuData ||
        tensor.ort_tensor?.data;

    if (!floatArray) {
        throw new Error("Unable to extract Float32Array from tensor result.");
    }

    // Convert Float32Array → normal JS array
    return Array.from(floatArray);
}
function buildUserProfileText({ bio, interests = [] }) {
    const interestsText = interests.length
        ? `Their interests include ${interests.join(', ')}.`
        : "";
    return `User dating profile, Bio:${bio}. ${interestsText}`.trim();
}

// for text embedding I want a prompt generator that takes:
/*
{
  bio: "Adventurous traveler who loves photography and coffee.",
  interests: ["hiking", "art", "technology"]
} and returns something like:
"Adventurous traveler who loves photography and coffee. Their interests include hiking, art, technology." but it should be the same for all
*/
export async function generatePrompt({
    modelId, // = "Xenova/gpt2", // Or try: "HuggingFaceTB/SmolLM-135M-Instruct"
    inputText,
    maxLength = 200,
}) {
    try {
        // Initialize the text generation pipeline
        // console.time("Pipeline Gener Load");
        // const generator = await pipeline("text2text-generation", modelId);
        const generator = await pipeline("text-generation", modelId);
        // console.timeEnd("Pipeline Load");

        // Combine user bio and interests
        // console.log("\n🧠 Input to model:", inputText);
        const prompt = `${inputText}`;
        console.log("\n🧠 Input to model:", prompt);

        // Generate a prompt
        console.time("Text Generation");
        const output = await generator(prompt, {
            max_new_tokens: maxLength,
            temperature: 0.7, // creativity
            top_p: 0.9, // nucleus sampling
        });
        console.timeEnd("Text Generation");

        console.log("ouput:", output)

        // Output generated text
        let paragraph = output[0].generated_text
        // .split("Description:")[1]  // only keep the continuation
        // .trim();

        return paragraph;
    } catch (err) {
        console.error("❌ Error generating prompt:", err);
        throw err;
    }
}

// text embedding
// sentence-transformers/all-MiniLM-L6-v2 (384D, fast)
// sentence-transformers/all-mpnet-base-v2 (768D, better quality)
// BAAI/bge-small-en-v1.5 (384D, good performance)
// BAAI/bge-base-en-v1.5 (768D, high quality)
async function getTextEmbedding({
    modelId,
    params,
}) {
    try {
        console.time('Pipeline Text load');
        const extractor = await pipeline('feature-extraction', modelId);
        console.timeEnd('Pipeline Text load');
        console.time('Text embedding');
        const result = await extractor(params);
        console.timeEnd('Text embedding');
        return result;
    } catch (e) {
        console.error("Error extracting text embedding:", e);
        throw e;
    }
}

// image embedding
async function getImageEmbedding({ modelId, imageUrl }) {
    try {
        console.time("Pipeline Image load");
        const extractor = await pipeline("image-feature-extraction", modelId);
        console.timeEnd("Pipeline Image load");

        console.time("Image embedding");
        const result = await extractor(imageUrl);
        console.timeEnd("Image embedding");

        return result;
    } catch (error) {
        console.error("Error extracting image embedding:", error);
        throw error;
    }
}

console.time("Total Text time");
const exampleUser = {
    bio: "Curious traveler and coffee lover who enjoys photography and tech gadgets",
    interests: ["AI", "hiking", "books"],
};
const exampleUser2 = {
    bio: "A mom who loves to cook and hike, here to meet a long term partner",
    interests: ["Books", "hiking", "cooking"],
};
const userProfileText = buildUserProfileText(exampleUser2);
const generatedPromptText = await generatePrompt({
    // modelId: "Xenova/gpt2",
    // modelId: "HuggingFaceTB/SmolLM-135M-Instruct",
    modelId: "HuggingFaceTB/SmolLM2-360M-Instruct",
    // modelId: "google-t5/t5-small",
    inputText: userProfileText,
    maxLength: 400
});
console.timeEnd("Total Text time");
console.log("generatedPromptText:", generatedPromptText);

const textEmbeddingResult = await getTextEmbedding({
    modelId: "sentence-transformers/all-MiniLM-L6-v2",
    params: userProfileText,
});
console.log("Embedding Text result:", extractEmbedding(textEmbeddingResult));

const imageXenovaEmbeddingResult = await getImageEmbedding(
    {
        modelId: "Xenova/clip-vit-base-patch32",
        imageUrl: "https://blog.photofeeler.com/wp-content/uploads/2016/06/good-dating-app-selfie-woman.jpeg",
    }
);
console.log("imageXenovaEmbeddingResult: ", extractEmbedding(imageXenovaEmbeddingResult));
