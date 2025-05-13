import {
  AutoTokenizer,
  ClapTextModelWithProjection,
  AutoProcessor,
  ClapAudioModelWithProjection,
  read_audio,
  PreTrainedTokenizer,
  Processor,
} from "@huggingface/transformers";
//@ts-ignore
import { VectorDB } from "idb-vector";

// --- Constants ---
const TEXT_MODEL_NAME = "Xenova/clap-htsat-unfused";
const AUDIO_MODEL_NAME = "Xenova/clap-htsat-unfused";
const DB_NAME = "clap-embeddings-db";
const TAG_AUDIO = "audio";

// --- Model and DB Instances ---
let tokenizer: PreTrainedTokenizer | null = null;
let textModel: ClapTextModelWithProjection | null = null;
let audioProcessor: Processor | null = null;
let audioModel: ClapAudioModelWithProjection | null = null;
let db: VectorDB | null = null;

// --- UI Elements ---
const audioFolderInput = document.getElementById(
  "audioFolderInput"
) as HTMLInputElement;
const embedAudioButton = document.getElementById(
  "embedAudioButton"
) as HTMLButtonElement;
const audioEmbedStatus = document.getElementById(
  "audioEmbedStatus"
) as HTMLParagraphElement;

const textSearchInput = document.getElementById(
  "textSearchInput"
) as HTMLInputElement;
const searchButton = document.getElementById(
  "searchButton"
) as HTMLButtonElement;

const audioSearchInput = document.getElementById(
  "audioSearchInput"
) as HTMLInputElement;
const searchAudioButton = document.getElementById(
  "searchAudioButton"
) as HTMLButtonElement;

const searchResultsList = document.getElementById(
  "searchResultsList"
) as HTMLUListElement;
const clearDbButton = document.getElementById(
  "clearDbButton"
) as HTMLButtonElement;
const dbStatus = document.getElementById("dbStatus") as HTMLParagraphElement;
const modelStatus = document.getElementById(
  "modelStatus"
) as HTMLParagraphElement;

// --- Utility Functions ---
function updateModelStatus(message: string, loaded: boolean = false) {
  modelStatus.textContent = message;
  if (loaded) {
    modelStatus.style.color = "green";
  }
  console.log(message);
}

function updateAudioEmbedStatus(message: string) {
  audioEmbedStatus.textContent = message;
  console.log(message);
}

function updateDbStatus(message: string) {
  dbStatus.textContent = message;
  console.log(message);
}

function disableControls(disabled: boolean) {
  embedAudioButton.disabled = disabled;
  searchButton.disabled = disabled;
  searchAudioButton.disabled = disabled;
  clearDbButton.disabled = disabled;
  textSearchInput.disabled = disabled;
  audioFolderInput.disabled = disabled;
  audioSearchInput.disabled = disabled;
}

// --- Core Logic: Model Loading ---
async function loadModels() {
  disableControls(true);
  updateModelStatus("Loading tokenizer...");
  try {
    tokenizer = await AutoTokenizer.from_pretrained(TEXT_MODEL_NAME);
    updateModelStatus("Tokenizer loaded. Loading text model...");
    textModel = await ClapTextModelWithProjection.from_pretrained(
      TEXT_MODEL_NAME,
      {
        progress_callback: (progress: any) => {
          // console.log("PROGRESS:", progress);
          updateModelStatus(
            `Loading text model: ${progress.file} (${Math.round(
              progress.progress
            )}%)...`
          );
        },
      }
    );
    updateModelStatus("Text model loaded. Loading audio processor...");
    audioProcessor = await AutoProcessor.from_pretrained(AUDIO_MODEL_NAME, {});
    updateModelStatus("Audio processor loaded. Loading audio model...");
    audioModel = await ClapAudioModelWithProjection.from_pretrained(
      AUDIO_MODEL_NAME,
      {
        progress_callback: (progress: any) => {
          updateModelStatus(
            `Loading audio model: ${progress.file} (${Math.round(
              progress.progress
            )}%)...`
          );
        },
      }
    );
    updateModelStatus("All models loaded successfully!", true);
    disableControls(false); // Enable controls after models are loaded
  } catch (error) {
    updateModelStatus(`Error loading models: ${error}`, false);
    console.error("Model loading error:", error);
    disableControls(true);
  }
}

// --- Core Logic: Vector DB Initialization ---
async function initVectorDB() {
  disableControls(true);
  updateDbStatus("Initializing vector database...");
  try {
    db = new VectorDB({
      vectorPath: "embedding",
      dbName: DB_NAME,
    });
    updateDbStatus("Vector database initialized."); // Simplified message
    disableControls(false);
  } catch (error) {
    console.warn(
      "Db constructor failed. Attempting Db.new() as fallback",
      error
    );
    updateDbStatus(
      "Failed to initialize vector database. App will not function correctly."
    );
    disableControls(true);
  }
}

// --- Core Logic: Embeddings ---
async function getTextEmbedding(text: string): Promise<Float32Array | null> {
  if (!tokenizer || !textModel) {
    updateModelStatus("Text model or tokenizer not loaded.", false);
    return null;
  }
  try {
    const text_inputs = tokenizer(text, { padding: true, truncation: true });
    const { text_embeds } = await textModel(text_inputs);
    return text_embeds.data as Float32Array;
  } catch (error) {
    console.error("Error getting text embedding:", error);
    updateModelStatus(`Error creating text embedding: ${error}`, false);
    return null;
  }
}

async function getAudioEmbedding(
  audioFile: File
): Promise<Float32Array | null> {
  if (!audioProcessor || !audioModel) {
    updateModelStatus("Audio model or processor not loaded.", false);
    return null;
  }
  let objectUrl: string | null = null;
  try {
    // Create an object URL for the local file
    objectUrl = URL.createObjectURL(audioFile);

    // read_audio now takes the URL. Sampling rate should be inferred from the file.
    const audioData = await read_audio(objectUrl, 44100);

    const audio_inputs = await audioProcessor(audioData);
    const { audio_embeds } = await audioModel(audio_inputs);
    return audio_embeds.data as Float32Array;
  } catch (error) {
    console.error("Error getting audio embedding:", error);
    updateModelStatus(`Error creating audio embedding: ${error}`, false);
    return null;
  } finally {
    // Revoke the object URL to free up resources
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
    }
  }
}

// --- Core Logic: Database Operations ---
async function insertIntoDB(
  content: string,
  embedding: Float32Array,
  tags: string[] = []
) {
  if (!db) {
    updateDbStatus("Database not initialized.");
    return;
  }
  try {
    // const float64Embedding = new Float64Array(embedding); // Convert F32 to F64 for Victor
    const array = Array.from(embedding);
    await db.insert({ content, embedding: array, tags });
    console.log(`Inserted: ${content}`);
  } catch (error) {
    updateDbStatus(`Error inserting into DB: ${error}`);
    console.error("DB insert error:", error);
  }
}

async function searchDB(
  embedding: Float32Array,
  k: number = 10
): Promise<
  {
    object: { content: string; tags: string[]; embedding: number[] };
    similarity: number;
    key: number;
  }[]
> {
  if (!db) {
    updateDbStatus("Database not initialized.");
    return [];
  }
  if (!embedding || embedding.length === 0) {
    updateModelStatus("Cannot search with empty embedding.", false);
    return [];
  }
  try {
    const array = Array.from(embedding);
    const results: {
      object: { content: string; tags: string[]; embedding: number[] };
      similarity: number;
      key: number;
    }[] = await db.query(array, {
      limit: k,
    });
    console.log("Search results:", results);
    return results;
  } catch (error) {
    updateDbStatus(`Error searching DB: ${error}`);
    console.error("DB search error:", error);
    return [];
  }
}

async function clearDB() {
  if (!db) {
    updateDbStatus("Database not initialized.");
    console.error("DB does not exist");
    return;
  }
  // await db.clear();
  const request = indexedDB.deleteDatabase(DB_NAME);
  request.onerror = (e) => {
    //@ts-ignore
    updateDbStatus(`Error clearing DB: ${e.target.error}`);
  };
  request.onblocked = () => {
    alert("Database clearing was blocked, probably due to an open connection.");
  };
  request.onsuccess = (e) => {
    console.log("Database cleared");
    updateDbStatus("Database cleared."); // Simplified message
    searchResultsList.innerHTML = ""; // Clear results display
  };
}

// --- UI Event Handlers ---
async function handleEmbedAudioFolder() {
  if (!audioFolderInput.files || audioFolderInput.files.length === 0) {
    updateAudioEmbedStatus("No folder/files selected.");
    return;
  }
  if (!audioModel || !audioProcessor || !db) {
    updateAudioEmbedStatus("Models or DB not ready.");
    return;
  }

  disableControls(true);
  updateAudioEmbedStatus(
    `Processing ${audioFolderInput.files.length} files...`
  );
  let count = 0;
  const files = Array.from(audioFolderInput.files);

  for (const file of files) {
    if (file.type.startsWith("audio/")) {
      try {
        updateAudioEmbedStatus(
          `Embedding ${file.name}... (${count + 1}/${files.length})`
        );
        const embedding = await getAudioEmbedding(file);
        if (embedding) {
          await insertIntoDB(file.name, embedding, [TAG_AUDIO, file.type]);
          count++;
        }
      } catch (err) {
        updateAudioEmbedStatus(`Error embedding ${file.name}: ${err}`);
        console.error(`Error embedding ${file.name}:`, err);
      }
    } else {
      updateAudioEmbedStatus(`Skipping non-audio file: ${file.name}`);
    }
  }
  updateAudioEmbedStatus(`Finished embedding ${count} audio files.`);
  disableControls(false);
}

async function handleTextSearch() {
  const query = textSearchInput.value.trim();
  if (!query) {
    alert("Please enter a search query.");
    return;
  }
  if (!textModel || !tokenizer || !db) {
    updateModelStatus("Models or DB not ready for search.", false);
    return;
  }

  disableControls(true);
  updateModelStatus("Generating text embedding for search...", false);
  const embedding = await getTextEmbedding(query);
  if (embedding) {
    updateModelStatus("Searching database...", false);
    const results = await searchDB(embedding, 10);
    displayResults(results, "text", query);
    updateModelStatus("Search complete.", true);
  } else {
    updateModelStatus("Failed to generate embedding for search.", false);
  }
  disableControls(false);
}

async function handleAudioFileSearch() {
  if (!audioSearchInput.files || audioSearchInput.files.length === 0) {
    alert("Please select an audio file to search with.");
    return;
  }
  const audioFile = audioSearchInput.files[0];
  if (!audioModel || !audioProcessor || !db) {
    updateModelStatus("Models or DB not ready for search.", false);
    return;
  }

  disableControls(true);
  updateModelStatus(
    `Generating audio embedding for ${audioFile.name}...`,
    false
  );
  const embedding = await getAudioEmbedding(audioFile);
  if (embedding) {
    updateModelStatus("Searching database with audio query...", false);
    // Search for other audio files. Could also search for text if we stored text with audio tags.
    const results = await searchDB(embedding, 10);
    displayResults(results, "audio", audioFile.name);
    updateModelStatus("Audio search complete.", true);
  } else {
    updateModelStatus("Failed to generate embedding for audio search.", false);
  }
  disableControls(false);
}

function displayResults(
  results: {
    object: { content: string; tags: string[]; embedding: number[] };
    similarity: number;
    key: number;
  }[],
  type: "text" | "audio",
  query: string
) {
  searchResultsList.innerHTML = ""; // Clear previous results

  if (results.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No results found.";
    searchResultsList.appendChild(li);
    return;
  }

  const header = document.createElement("p");
  header.innerHTML = `Showing ${results.length} results for ${type} query: "<strong>${query}</strong>"`;
  searchResultsList.appendChild(header);

  results.forEach((result) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${
      result.object.content
    }</strong> (Similarity Score: ${result.similarity.toFixed(4)})`;
    searchResultsList.appendChild(li);
  });
}

// --- Initialization ---
async function main() {
  disableControls(true); // Disable all controls until models are loaded
  await initVectorDB();
  await loadModels();
  // Only enable controls if both DB and Models loaded successfully.
  if (db && tokenizer && textModel && audioProcessor && audioModel) {
    disableControls(false);
  } else {
    updateModelStatus(
      "Application initialization failed. Some features may be disabled.",
      false
    );
  }
}

// --- Event Listeners ---
embedAudioButton.addEventListener("click", handleEmbedAudioFolder);
searchButton.addEventListener("click", handleTextSearch);
searchAudioButton.addEventListener("click", handleAudioFileSearch);
clearDbButton.addEventListener("click", clearDB);

// Start the application
main().catch(console.error);
