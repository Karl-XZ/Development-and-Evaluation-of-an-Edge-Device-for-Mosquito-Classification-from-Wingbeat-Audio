#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <SPI.h>
#include <SdFat.h>
#include <new>
#include "driver/i2s.h"
#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "models/model_overlap025.h"
#include "models/model_overlap025_logmel.h"

constexpr gpio_num_t kMicSckPin = GPIO_NUM_5;  // D2
constexpr gpio_num_t kMicWsPin = GPIO_NUM_6;   // D3
constexpr gpio_num_t kMicSdPin = GPIO_NUM_7;   // D4
constexpr i2s_port_t kI2sPort = I2S_NUM_0;
constexpr int32_t kMicGain = 4;

// SD card: D10=CS, D11=MOSI, D12=MISO, D13=SCK.  These are the same pins and
// conservative 1 MHz SPI clock used by the previously verified S/E recorder.
constexpr int kSdCsPin = D10;
constexpr int kSdMosiPin = D11;
constexpr int kSdMisoPin = D12;
constexpr int kSdSckPin = D13;
SdFs sd;
SdSpiConfig sdConfig(kSdCsPin, SHARED_SPI, SD_SCK_MHZ(1));

constexpr int kRecorderRate = 16000;
constexpr int kContextRecorderSamples = 24000;  // 1.5 seconds
constexpr int kModelSamples = 12000;            // 1.5 seconds at 8 kHz
constexpr int kFftSize = 256;
constexpr int kFftBins = 129;
constexpr int kFrames = 151;
constexpr int kMels = 64;
constexpr int kFeatureCount = kMels * kFrames;
constexpr int kCenterPadding = 128;
constexpr size_t kTensorArenaBytes = 640 * 1024;
constexpr size_t kAudioRingSamples = 131072;  // 8.19 seconds at 16 kHz
constexpr int kStandardScoreHistory = 3;
constexpr char kFirmwareBuild[] = "lab_realtime_recorder_v005_dual_frontend_exact3s";
constexpr char kLogModelBuild[] = "logmel_selected_fold_int8";
constexpr char kPcenModelBuild[] = "pcen_selected_fold_int8";
constexpr char kSpeedPatchBuild[] = "cmsis_nn_verified_3file";
constexpr float kPcenAlpha = 0.60f;
constexpr float kPcenDelta = 2.0f;
constexpr float kPcenR = 0.25f;
constexpr float kPcenEps = 1.0e-6f;
// 1-exp(-hop/(sample_rate*time_constant)), hop=80, rate=8000, tau=0.8 s.
constexpr float kPcenSmoothing = 0.012422199506f;
constexpr float kPcenDeltaCompressed = 1.189207115003f;
constexpr uint32_t kAutoStartDelayMs = 10000;
constexpr uint32_t kI2sSlotDetectionFrames = kRecorderRate;
constexpr size_t kRecordingBlockSamples = 512;
constexpr int kRecordingQueueBlocks = 48;  // 1.536 s of buffering at 16 kHz.
constexpr uint32_t kRecordingSyncSamples = 5 * kRecorderRate;
constexpr uint64_t kMaximumRecordingSamples = 3600ULL * kRecorderRate;
// Exact PC research geometry for a 3.0 s decision. Each endpoint below closes
// a 1.5 s context; the corresponding context centers are 0.75, 1.50, 2.25 s.
constexpr int kExactDecisionRecorderSamples = 3 * kRecorderRate;
constexpr int kExactContextEndOffsets[3] = {
  24000,  // 1.50 s
  36000,  // 2.25 s
  48000,  // 3.00 s
};

constexpr char kClassNames[][16] = {
  "ae_aegypti", "cx_quin", "an_dirus",
  "an_minimus_f", "an_minimus_m", "no_mosquito"
};

constexpr uint8_t kMelBins[kMels + 2] = {
  0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19,
  20, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 38, 40, 42, 43, 45, 48,
  50, 52, 54, 56, 59, 61, 64, 66, 69, 71, 74, 77, 80, 83, 86, 90, 93, 96,
  100, 104, 107, 111, 115, 119, 124, 128
};

struct ModelSpec {
  const unsigned char* data;
  unsigned int bytes;
  const char* name;
  int classes;
};

struct RecordingBlock {
  uint16_t count;
  int16_t samples[kRecordingBlockSamples];
};

const ModelSpec kModels[] = {
  {model_overlap025, model_overlap025_len, "original_alias_0", 6},
  {model_overlap025, model_overlap025_len, "original_alias_1", 6},
  {model_overlap025, model_overlap025_len, "original_overlap025", 6},
  {model_overlap025, model_overlap025_len, "original_alias_3", 6},
  {model_overlap025, model_overlap025_len, "original_alias_4", 6},
  {model_overlap025, model_overlap025_len, "original_alias_5", 6},
  {model_overlap025, model_overlap025_len, "original_alias_6", 6},
  {model_overlap025, model_overlap025_len, "original_alias_7", 6},
  {model_overlap025, model_overlap025_len, "original_alias_8", 6},
  {model_overlap025_logmel, model_overlap025_logmel_len, "original_logmel", 6},
};

// Power-on default is log-Mel. A/B select a matched frontend and model pair.
bool usePcen = false;
const char* activeFrontendName() { return usePcen ? "pcen" : "logmel"; }
const char* activeModelName() { return usePcen ? kPcenModelBuild : kLogModelBuild; }
int activeModelIndex() { return usePcen ? 2 : 9; }

int16_t* recorderSamples = nullptr;
float* modelSamples = nullptr;
float* logMelFeatures = nullptr;
uint8_t* tensorArena = nullptr;
int32_t* i2sWords = nullptr;
int16_t* audioRing = nullptr;
float fftReal[kFftSize];
float fftImag[kFftSize];
float hannWindow[kFftSize];
float twiddleReal[kFftSize / 2];
float twiddleImag[kFftSize / 2];

// Every bundled model has the same four-op CNN graph.  Keeping the resolver
// exact avoids compiling dozens of unused reference kernels and permits the
// Xtensa toolchain to optimize the kernels that are actually executed.
tflite::MicroMutableOpResolver<4> resolver;
alignas(tflite::MicroInterpreter) uint8_t interpreterStorage[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;
int loadedModel = -1;
bool microphoneReady = false;
bool liveMode = false;
int selectedMode = 0;
bool contextReady = false;
uint32_t resultIndex = 0;
uint32_t lastCaptureMs = 0;
uint32_t lastFeatureMs = 0;
uint32_t lastInferenceMs = 0;
uint32_t currentResultStartMs = 0;
float calibrationThreshold[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
bool calibrationValid[5] = {false, false, false, false, false};
bool awaitingCalibrationMode = false;
int pendingCalibrationMode = 0;
uint32_t pendingCalibrationSinceMs = 0;
bool liveUseCalibration = false;
float lastClassScores[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
uint32_t skippedHopCount = 0;
uint64_t nextAudioEndSample = 0;
uint64_t lastDecisionEndSample = 0;
volatile uint64_t audioSamplesWritten = 0;
volatile bool audioCaptureError = false;
volatile int activeI2sWordOffset = 0;
volatile bool i2sSlotDetectionComplete = false;
TaskHandle_t audioCaptureTaskHandle = nullptr;
portMUX_TYPE audioRingMux = portMUX_INITIALIZER_UNLOCKED;

bool sdReady = false;
volatile bool recordingActive = false;
volatile bool recordingError = false;
volatile uint32_t recordingBlocksQueued = 0;
volatile uint32_t recordingBlocksWritten = 0;
volatile uint32_t recordingDroppedBlocks = 0;
volatile uint64_t recordingFirstSample = UINT64_MAX;
uint32_t recordingSamplesWritten = 0;
uint32_t recordingDataBytes = 0;
uint32_t recordingPeak = 0;
uint32_t lastRecordingSyncSample = 0;
uint32_t sessionStartedMs = 0;
char sessionBaseName[7] = {0};
char wavFilename[13] = {0};
char csvFilename[13] = {0};
bool autoStartPending = false;
uint32_t autoStartDeadlineMs = 0;
bool awaitingSessionSelection = false;
char sessionSelectionDigits[4] = {0};
uint8_t sessionSelectionDigitCount = 0;
FsFile wavFile;
FsFile csvFile;
QueueHandle_t recordingQueue = nullptr;
TaskHandle_t sdWriterTaskHandle = nullptr;
SemaphoreHandle_t sdMutex = nullptr;

void printStatus();
bool pollStopDuringWork();
void stopLiveSession(const char* reason);

void sendUint32LE(uint32_t value) {
  uint8_t bytes[4] = {
    static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
    static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24)
  };
  Serial.write(bytes, 4);
}

bool readSerialExact(uint8_t* destination, size_t bytes, uint32_t timeoutMs) {
  size_t received = 0;
  uint32_t deadline = millis() + timeoutMs;
  while (received < bytes) {
    const int available = Serial.available();
    if (available > 0) {
      const size_t wanted = min(static_cast<size_t>(available), bytes - received);
      const size_t count = Serial.readBytes(
          reinterpret_cast<char*>(destination + received), wanted);
      received += count;
      deadline = millis() + timeoutMs;
    } else {
      if (static_cast<int32_t>(millis() - deadline) >= 0) return false;
      delay(1);
    }
  }
  return true;
}

void writeLE16(FsFile& file, uint16_t value) {
  const uint8_t bytes[2] = {
    static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8)
  };
  file.write(bytes, sizeof(bytes));
}

void writeLE32(FsFile& file, uint32_t value) {
  const uint8_t bytes[4] = {
    static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
    static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24)
  };
  file.write(bytes, sizeof(bytes));
}

bool writeWavHeader(FsFile& file, uint32_t dataBytes) {
  constexpr uint16_t channels = 1;
  constexpr uint16_t bitsPerSample = 16;
  const uint32_t byteRate = kRecorderRate * channels * bitsPerSample / 8;
  const uint16_t blockAlign = channels * bitsPerSample / 8;
  if (!file.seekSet(0)) return false;
  file.write(reinterpret_cast<const uint8_t*>("RIFF"), 4);
  writeLE32(file, 36 + dataBytes);
  file.write(reinterpret_cast<const uint8_t*>("WAVE"), 4);
  file.write(reinterpret_cast<const uint8_t*>("fmt "), 4);
  writeLE32(file, 16);
  writeLE16(file, 1);
  writeLE16(file, channels);
  writeLE32(file, kRecorderRate);
  writeLE32(file, byteRate);
  writeLE16(file, blockAlign);
  writeLE16(file, bitsPerSample);
  file.write(reinterpret_cast<const uint8_t*>("data"), 4);
  writeLE32(file, dataBytes);
  return !file.getWriteError();
}

bool initialiseSDCard() {
  SPI.begin(kSdSckPin, kSdMisoPin, kSdMosiPin, kSdCsPin);
  pinMode(kSdCsPin, OUTPUT);
  digitalWrite(kSdCsPin, HIGH);
  delay(100);
  if (!sd.cardBegin(sdConfig)) return false;
  return sd.volumeBegin();
}

bool chooseSessionFilenames() {
  for (uint16_t index = 1; index <= 999; ++index) {
    snprintf(sessionBaseName, sizeof(sessionBaseName), "LAB%03u", index);
    snprintf(wavFilename, sizeof(wavFilename), "%s.WAV", sessionBaseName);
    snprintf(csvFilename, sizeof(csvFilename), "%s.CSV", sessionBaseName);
    if (!sd.exists(wavFilename) && !sd.exists(csvFilename)) return true;
  }
  return false;
}

bool selectSessionFiles(uint16_t index) {
  if (index < 1 || index > 999) return false;
  char candidateBase[7] = {0};
  char candidateWav[13] = {0};
  char candidateCsv[13] = {0};
  snprintf(candidateBase, sizeof(candidateBase), "LAB%03u", index);
  snprintf(candidateWav, sizeof(candidateWav), "%s.WAV", candidateBase);
  snprintf(candidateCsv, sizeof(candidateCsv), "%s.CSV", candidateBase);
  if (!sd.exists(candidateWav) || !sd.exists(candidateCsv)) return false;
  strncpy(sessionBaseName, candidateBase, sizeof(sessionBaseName) - 1);
  sessionBaseName[sizeof(sessionBaseName) - 1] = '\0';
  strncpy(wavFilename, candidateWav, sizeof(wavFilename) - 1);
  wavFilename[sizeof(wavFilename) - 1] = '\0';
  strncpy(csvFilename, candidateCsv, sizeof(csvFilename) - 1);
  csvFilename[sizeof(csvFilename) - 1] = '\0';
  return true;
}

void printSessionList() {
  if (liveMode || recordingActive || !sdReady) {
    Serial.println("SESSION_LIST_ERROR stop_live_mode_first");
    return;
  }
  Serial.println("SESSION_LIST_BEGIN");
  uint16_t count = 0;
  uint8_t consecutiveMissing = 0;
  for (uint16_t index = 1; index <= 999; ++index) {
    char candidateWav[13] = {0};
    char candidateCsv[13] = {0};
    snprintf(candidateWav, sizeof(candidateWav), "LAB%03u.WAV", index);
    snprintf(candidateCsv, sizeof(candidateCsv), "LAB%03u.CSV", index);
    if (!sd.exists(candidateWav) && !sd.exists(candidateCsv)) {
      if (count > 0 && ++consecutiveMissing >= 16) break;
      continue;
    }
    consecutiveMissing = 0;
    FsFile wav;
    FsFile csv;
    uint32_t wavBytes = 0;
    uint32_t csvBytes = 0;
    if (wav.open(candidateWav, O_RDONLY)) {
      wavBytes = static_cast<uint32_t>(wav.fileSize());
      wav.close();
    }
    if (csv.open(candidateCsv, O_RDONLY)) {
      csvBytes = static_cast<uint32_t>(csv.fileSize());
      csv.close();
    }
    Serial.print("SESSION name=LAB");
    if (index < 100) Serial.print('0');
    if (index < 10) Serial.print('0');
    Serial.print(index);
    Serial.print(" wav_bytes="); Serial.print(wavBytes);
    Serial.print(" csv_bytes="); Serial.println(csvBytes);
    ++count;
  }
  Serial.print("SESSION_LIST_END count="); Serial.println(count);
}

bool repairSelectedWavHeader() {
  if (!wavFilename[0] || !sd.exists(wavFilename)) return false;
  FsFile file;
  if (!file.open(wavFilename, O_RDWR)) return false;
  uint32_t fileBytes = static_cast<uint32_t>(file.fileSize());
  if (fileBytes < 44) {
    file.close();
    return false;
  }
  uint32_t dataBytes = fileBytes - 44;
  if (dataBytes & 1U) {
    --dataBytes;
    if (!file.truncate(44 + dataBytes)) {
      file.close();
      return false;
    }
  }
  const bool ok = writeWavHeader(file, dataBytes) && file.sync();
  file.close();
  return ok;
}

void sdWriterTask(void*) {
  RecordingBlock block;
  while (true) {
    if (xQueueReceive(recordingQueue, &block, portMAX_DELAY) != pdTRUE) continue;
    bool writeOk = false;
    if (sdMutex && xSemaphoreTake(sdMutex, pdMS_TO_TICKS(2000)) == pdTRUE) {
      const size_t bytes = static_cast<size_t>(block.count) * sizeof(int16_t);
      writeOk = wavFile.isOpen() &&
          wavFile.write(reinterpret_cast<const uint8_t*>(block.samples), bytes) == bytes;
      if (writeOk) {
        recordingSamplesWritten += block.count;
        recordingDataBytes += bytes;
        for (uint16_t index = 0; index < block.count; ++index) {
          const uint32_t amplitude = abs(static_cast<int32_t>(block.samples[index]));
          recordingPeak = max(recordingPeak, amplitude);
        }
        if (recordingSamplesWritten - lastRecordingSyncSample >= kRecordingSyncSamples) {
          writeOk = wavFile.sync();
          lastRecordingSyncSample = recordingSamplesWritten;
        }
      }
      xSemaphoreGive(sdMutex);
    }
    if (!writeOk) recordingError = true;
    ++recordingBlocksWritten;
  }
}

bool startRecordingSession(int mode, bool calibrated) {
  if (!sdReady || !recordingQueue || !sdMutex || !chooseSessionFilenames()) return false;
  xQueueReset(recordingQueue);
  recordingError = false;
  recordingBlocksQueued = 0;
  recordingBlocksWritten = 0;
  recordingDroppedBlocks = 0;
  recordingFirstSample = UINT64_MAX;
  recordingSamplesWritten = 0;
  recordingDataBytes = 0;
  recordingPeak = 0;
  lastRecordingSyncSample = 0;
  sessionStartedMs = millis();

  if (xSemaphoreTake(sdMutex, pdMS_TO_TICKS(2000)) != pdTRUE) return false;
  bool ok = wavFile.open(wavFilename, O_RDWR | O_CREAT | O_TRUNC);
  if (ok) ok = writeWavHeader(wavFile, 0);
  if (ok) ok = csvFile.open(csvFilename, O_WRONLY | O_CREAT | O_TRUNC);
  if (ok) {
    csvFile.print("# firmware="); csvFile.println(kFirmwareBuild);
    csvFile.print("# frontend="); csvFile.println(activeFrontendName());
    csvFile.print("# deployed_model="); csvFile.println(activeModelName());
    csvFile.print("# speed_patch="); csvFile.println(kSpeedPatchBuild);
    if (usePcen) csvFile.println("# pcen_alpha=0.6,pcen_delta=2.0,pcen_r=0.25,pcen_time_constant_sec=0.8,pcen_eps=0.000001");
    csvFile.println("# sample_rate_hz=16000,channels=1,bits_per_sample=16");
    csvFile.println("# maximum_recording_seconds=3600");
    csvFile.print("# mode="); csvFile.print(mode);
    csvFile.print(",calibrated="); csvFile.println(calibrated ? 1 : 0);
    csvFile.println("result_index,session_elapsed_ms,audio_end_ms,support_start_ms,support_end_ms,mode,class,ae_aegypti,cx_quin,an_dirus,an_minimus_f,an_minimus_m,no_mosquito,calibrated,no_margin,no_threshold,snapshot_wait_ms,feature_ms,infer_ms,audio_lag_ms,latency_ms,skipped_hops_total");
    ok = csvFile.sync() && wavFile.sync();
  }
  if (!ok) {
    if (csvFile.isOpen()) csvFile.close();
    if (wavFile.isOpen()) wavFile.close();
  }
  xSemaphoreGive(sdMutex);
  if (!ok) return false;

  recordingActive = true;
  const uint32_t deadline = millis() + 1500;
  while (recordingFirstSample == UINT64_MAX && static_cast<int32_t>(millis() - deadline) < 0) {
    delay(2);
  }
  if (recordingFirstSample == UINT64_MAX) {
    recordingActive = false;
    recordingError = true;
    return false;
  }
  return true;
}

void logPrediction(int mode, int predicted, float noMargin, uint32_t audioLagMs,
                   uint32_t latencyMs) {
  if (!recordingActive || recordingFirstSample == UINT64_MAX || !csvFile.isOpen()) return;
  const uint64_t relativeEndSamples = lastDecisionEndSample >= recordingFirstSample
      ? lastDecisionEndSample - recordingFirstSample : 0;
  const uint32_t audioEndMs = static_cast<uint32_t>(relativeEndSamples * 1000ULL / kRecorderRate);
  const uint32_t supportMs = mode == 1 ? 1500 : 3000;
  const uint32_t supportStartMs = audioEndMs > supportMs ? audioEndMs - supportMs : 0;
  if (xSemaphoreTake(sdMutex, pdMS_TO_TICKS(2000)) != pdTRUE) {
    recordingError = true;
    return;
  }
  csvFile.print(resultIndex); csvFile.print(',');
  csvFile.print(millis() - sessionStartedMs); csvFile.print(',');
  csvFile.print(audioEndMs); csvFile.print(',');
  csvFile.print(supportStartMs); csvFile.print(',');
  csvFile.print(audioEndMs); csvFile.print(',');
  csvFile.print(mode); csvFile.print(',');
  csvFile.print(kClassNames[predicted]);
  for (int index = 0; index < 6; ++index) {
    csvFile.print(','); csvFile.print(lastClassScores[index], 7);
  }
  csvFile.print(','); csvFile.print(liveUseCalibration ? 1 : 0);
  csvFile.print(','); csvFile.print(noMargin, 7);
  csvFile.print(','); csvFile.print(liveUseCalibration ? calibrationThreshold[mode] : 0.0f, 7);
  csvFile.print(','); csvFile.print(lastCaptureMs);
  csvFile.print(','); csvFile.print(lastFeatureMs);
  csvFile.print(','); csvFile.print(lastInferenceMs);
  csvFile.print(','); csvFile.print(audioLagMs);
  csvFile.print(','); csvFile.print(latencyMs);
  csvFile.print(','); csvFile.println(skippedHopCount);
  // Predictions are small and arrive once per completed 3.0 s decision. Sync every row
  // so a sudden power loss can discard at most the row being written.
  if (csvFile.getWriteError() || !csvFile.sync()) recordingError = true;
  xSemaphoreGive(sdMutex);
}

void stopRecordingSession() {
  if (!recordingActive && !wavFile.isOpen() && !csvFile.isOpen()) return;
  recordingActive = false;
  // Let a capture task that already observed recordingActive finish its final
  // queue copy before the queued/written counters are compared.
  delay(50);
  const uint32_t drainDeadline = millis() + 5000;
  while (recordingBlocksWritten < recordingBlocksQueued &&
         static_cast<int32_t>(millis() - drainDeadline) < 0) {
    delay(5);
  }
  if (recordingBlocksWritten != recordingBlocksQueued) recordingError = true;

  bool finalizeOk = false;
  if (xSemaphoreTake(sdMutex, pdMS_TO_TICKS(3000)) == pdTRUE) {
    const bool headerOk = wavFile.isOpen() && writeWavHeader(wavFile, recordingDataBytes);
    const bool wavSyncOk = headerOk && wavFile.sync();
    const bool csvSyncOk = csvFile.isOpen() && csvFile.sync();
    if (csvFile.isOpen()) csvFile.close();
    if (wavFile.isOpen()) wavFile.close();
    finalizeOk = wavSyncOk && csvSyncOk;
    xSemaphoreGive(sdMutex);
  }
  if (!finalizeOk) recordingError = true;

  Serial.print(recordingError ? "SESSION_RECORD_ERROR" : "SESSION_RECORD_DONE");
  Serial.print(" wav="); Serial.print(wavFilename);
  Serial.print(" csv="); Serial.print(csvFilename);
  Serial.print(" bytes="); Serial.print(recordingDataBytes + 44);
  Serial.print(" seconds="); Serial.print(static_cast<float>(recordingSamplesWritten) / kRecorderRate, 3);
  Serial.print(" peak="); Serial.print(recordingPeak);
  Serial.print(" queued_blocks="); Serial.print(recordingBlocksQueued);
  Serial.print(" written_blocks="); Serial.print(recordingBlocksWritten);
  Serial.print(" dropped_blocks="); Serial.println(recordingDroppedBlocks);
}

bool selectLatestSessionFiles() {
  if (wavFilename[0] && csvFilename[0] && sd.exists(wavFilename) && sd.exists(csvFilename)) {
    return true;
  }
  for (int index = 999; index >= 1; --index) {
    char candidateWav[13] = {0};
    char candidateCsv[13] = {0};
    snprintf(candidateWav, sizeof(candidateWav), "LAB%03d.WAV", index);
    snprintf(candidateCsv, sizeof(candidateCsv), "LAB%03d.CSV", index);
    if (sd.exists(candidateWav) && sd.exists(candidateCsv)) {
      strncpy(wavFilename, candidateWav, sizeof(wavFilename) - 1);
      strncpy(csvFilename, candidateCsv, sizeof(csvFilename) - 1);
      return true;
    }
  }
  return false;
}

void sendLatestSessionFile(bool sendCsv) {
  if (liveMode || recordingActive || !sdReady || !selectLatestSessionFiles()) {
    Serial.write(reinterpret_cast<const uint8_t*>("ERR1"), 4);
    sendUint32LE(0);
    Serial.flush();
    return;
  }
  if (xSemaphoreTake(sdMutex, pdMS_TO_TICKS(3000)) != pdTRUE) {
    Serial.write(reinterpret_cast<const uint8_t*>("ERR1"), 4);
    sendUint32LE(0);
    Serial.flush();
    return;
  }
  if (!sendCsv && !repairSelectedWavHeader()) {
    xSemaphoreGive(sdMutex);
    Serial.write(reinterpret_cast<const uint8_t*>("ERR1"), 4);
    sendUint32LE(0);
    Serial.flush();
    return;
  }
  FsFile file;
  const char* filename = sendCsv ? csvFilename : wavFilename;
  if (!file.open(filename, O_RDONLY)) {
    xSemaphoreGive(sdMutex);
    Serial.write(reinterpret_cast<const uint8_t*>("ERR1"), 4);
    sendUint32LE(0);
    Serial.flush();
    return;
  }
  const uint32_t fileSize = static_cast<uint32_t>(file.fileSize());
  Serial.write(reinterpret_cast<const uint8_t*>(sendCsv ? "CSV1" : "WAV1"), 4);
  sendUint32LE(fileSize);
  uint8_t buffer[1024];
  uint32_t sent = 0;
  while (sent < fileSize) {
    const size_t wanted = min(sizeof(buffer), static_cast<size_t>(fileSize - sent));
    const int count = file.read(buffer, wanted);
    if (count <= 0) break;
    size_t offset = 0;
    while (offset < static_cast<size_t>(count)) {
      const size_t written = Serial.write(buffer + offset, count - offset);
      if (written == 0) delay(1);
      else offset += written;
    }
    sent += count;
  }
  Serial.flush();
  file.close();
  xSemaphoreGive(sdMutex);
}

bool allocateBuffers() {
  recorderSamples = static_cast<int16_t*>(heap_caps_malloc(
      kContextRecorderSamples * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  modelSamples = static_cast<float*>(heap_caps_malloc(
      kModelSamples * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  logMelFeatures = static_cast<float*>(heap_caps_malloc(
      kFeatureCount * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  tensorArena = static_cast<uint8_t*>(heap_caps_malloc(
      kTensorArenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  i2sWords = static_cast<int32_t*>(heap_caps_malloc(1024 * sizeof(int32_t), MALLOC_CAP_8BIT));
  audioRing = static_cast<int16_t*>(heap_caps_malloc(
      kAudioRingSamples * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  return recorderSamples && modelSamples && logMelFeatures && tensorArena && i2sWords && audioRing;
}

bool initialiseMicrophone() {
  i2s_config_t config = {};
  config.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
  config.sample_rate = kRecorderRate;
  config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  config.channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT;
  config.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  config.dma_buf_count = 12;
  config.dma_buf_len = 256;
  config.use_apll = false;
  config.tx_desc_auto_clear = false;
  config.fixed_mclk = 0;
  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = kMicSckPin;
  pins.ws_io_num = kMicWsPin;
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = kMicSdPin;
  esp_err_t result = i2s_driver_install(kI2sPort, &config, 0, nullptr);
  if (result != ESP_OK) return false;
  result = i2s_set_pin(kI2sPort, &pins);
  if (result != ESP_OK) {
    i2s_driver_uninstall(kI2sPort);
    return false;
  }
  i2s_zero_dma_buffer(kI2sPort);
  return true;
}

int16_t pcm16(int32_t rawWord) {
  int64_t value = (static_cast<int64_t>(rawWord >> 8) * kMicGain) >> 8;
  return static_cast<int16_t>(constrain(value, -32768LL, 32767LL));
}

uint64_t capturedSampleCount() {
  portENTER_CRITICAL(&audioRingMux);
  const uint64_t value = audioSamplesWritten;
  portEXIT_CRITICAL(&audioRingMux);
  return value;
}

void audioCaptureTask(void*) {
  int16_t converted[512];
  uint64_t slotEnergy[2] = {0, 0};
  uint32_t slotDetectionFrames = 0;
  while (true) {
    size_t bytesRead = 0;
    const esp_err_t result = i2s_read(
        kI2sPort, i2sWords, 1024 * sizeof(int32_t), &bytesRead, pdMS_TO_TICKS(1000));
    if (result != ESP_OK || bytesRead == 0) {
      audioCaptureError = true;
      vTaskDelay(pdMS_TO_TICKS(5));
      continue;
    }
    const size_t words = bytesRead / sizeof(int32_t);
    size_t count = 0;
    for (size_t index = 0; index + 1 < words; index += 2) {
      if (!i2sSlotDetectionComplete) {
        for (int slot = 0; slot < 2; ++slot) {
          const int64_t value = static_cast<int64_t>(i2sWords[index + slot]);
          slotEnergy[slot] += static_cast<uint64_t>(value < 0 ? -value : value);
        }
        ++slotDetectionFrames;
        if (slotDetectionFrames >= kI2sSlotDetectionFrames) {
          activeI2sWordOffset = slotEnergy[1] > slotEnergy[0] ? 1 : 0;
          i2sSlotDetectionComplete = true;
          Serial.print("I2S_SLOT_SELECTED offset="); Serial.print(activeI2sWordOffset);
          Serial.print(" energy0="); Serial.print(static_cast<uint32_t>(slotEnergy[0] >> 16));
          Serial.print(" energy1="); Serial.println(static_cast<uint32_t>(slotEnergy[1] >> 16));
        }
      }
      converted[count++] = pcm16(i2sWords[index + activeI2sWordOffset]);
    }
    uint64_t blockStartSample = 0;
    portENTER_CRITICAL(&audioRingMux);
    blockStartSample = audioSamplesWritten;
    for (size_t index = 0; index < count; ++index) {
      audioRing[(audioSamplesWritten + index) % kAudioRingSamples] = converted[index];
    }
    if (recordingActive && recordingFirstSample == UINT64_MAX) {
      recordingFirstSample = blockStartSample;
    }
    audioSamplesWritten += count;
    portEXIT_CRITICAL(&audioRingMux);

    if (recordingActive) {
      const uint64_t firstSample = recordingFirstSample;
      const uint64_t elapsedSamples = blockStartSample >= firstSample
          ? blockStartSample - firstSample : 0;
      const uint64_t remainingSamples = elapsedSamples < kMaximumRecordingSamples
          ? kMaximumRecordingSamples - elapsedSamples : 0;
      if (remainingSamples == 0) continue;
      RecordingBlock block = {};
      block.count = static_cast<uint16_t>(min(
          min(count, kRecordingBlockSamples), static_cast<size_t>(remainingSamples)));
      memcpy(block.samples, converted, block.count * sizeof(int16_t));
      if (xQueueSend(recordingQueue, &block, 0) == pdTRUE) {
        ++recordingBlocksQueued;
      } else {
        ++recordingDroppedBlocks;
        recordingError = true;
      }
    }
  }
}

bool startAudioCaptureTask() {
  audioSamplesWritten = 0;
  audioCaptureError = false;
  return xTaskCreatePinnedToCore(
             audioCaptureTask, "continuous_i2s", 4096, nullptr, 2,
             &audioCaptureTaskHandle, 0) == pdPASS;
}

bool snapshotContextAt(uint64_t endSample, bool allowLiveStop) {
  const uint32_t started = millis();
  while (capturedSampleCount() < endSample) {
    if (audioCaptureError) return false;
    if (allowLiveStop && pollStopDuringWork()) return false;
    delay(2);
  }
  portENTER_CRITICAL(&audioRingMux);
  const uint64_t written = audioSamplesWritten;
  if (endSample < kContextRecorderSamples ||
      written - endSample > kAudioRingSamples - kContextRecorderSamples) {
    portEXIT_CRITICAL(&audioRingMux);
    return false;
  }
  const uint64_t start = endSample - kContextRecorderSamples;
  for (int index = 0; index < kContextRecorderSamples; ++index) {
    recorderSamples[index] = audioRing[(start + index) % kAudioRingSamples];
  }
  portEXIT_CRITICAL(&audioRingMux);
  lastCaptureMs = millis() - started;
  lastDecisionEndSample = endSample;
  contextReady = true;
  return true;
}

bool captureSamples(int16_t* destination, size_t count) {
  size_t collected = 0;
  while (collected < count) {
    size_t bytesRead = 0;
    esp_err_t result = i2s_read(kI2sPort, i2sWords, 1024 * sizeof(int32_t),
                                &bytesRead, pdMS_TO_TICKS(1000));
    if (result != ESP_OK || bytesRead == 0) return false;
    const size_t words = bytesRead / sizeof(int32_t);
    for (size_t index = 0; index + 1 < words && collected < count; index += 2) {
      destination[collected++] = pcm16(i2sWords[index]);
    }
  }
  return true;
}

bool refreshContext(int stepRecorderSamples) {
  const uint32_t started = millis();
  if (!contextReady) {
    i2s_zero_dma_buffer(kI2sPort);
    if (!captureSamples(recorderSamples, kContextRecorderSamples)) return false;
    contextReady = true;
  } else {
    memmove(recorderSamples, recorderSamples + stepRecorderSamples,
            (kContextRecorderSamples - stepRecorderSamples) * sizeof(int16_t));
    if (!captureSamples(recorderSamples + kContextRecorderSamples - stepRecorderSamples,
                        stepRecorderSamples)) return false;
  }
  lastCaptureMs = millis() - started;
  return true;
}

void initialiseFft() {
  for (int i = 0; i < kFftSize; ++i) hannWindow[i] = 0.5f - 0.5f * cosf(2.0f * PI * i / kFftSize);
  for (int i = 0; i < kFftSize / 2; ++i) {
    const float angle = -2.0f * PI * i / kFftSize;
    twiddleReal[i] = cosf(angle);
    twiddleImag[i] = sinf(angle);
  }
}

void fft256() {
  for (int index = 1, reversed = 0; index < kFftSize; ++index) {
    int bit = kFftSize >> 1;
    for (; reversed & bit; bit >>= 1) reversed ^= bit;
    reversed ^= bit;
    if (index < reversed) {
      const float real = fftReal[index], imag = fftImag[index];
      fftReal[index] = fftReal[reversed]; fftImag[index] = fftImag[reversed];
      fftReal[reversed] = real; fftImag[reversed] = imag;
    }
  }
  for (int length = 2; length <= kFftSize; length <<= 1) {
    const int half = length >> 1, step = kFftSize / length;
    for (int start = 0; start < kFftSize; start += length) {
      for (int offset = 0; offset < half; ++offset) {
        const int twiddle = offset * step;
        const float real = fftReal[start + offset + half];
        const float imag = fftImag[start + offset + half];
        const float tr = real * twiddleReal[twiddle] - imag * twiddleImag[twiddle];
        const float ti = real * twiddleImag[twiddle] + imag * twiddleReal[twiddle];
        const float br = fftReal[start + offset], bi = fftImag[start + offset];
        fftReal[start + offset] = br + tr; fftImag[start + offset] = bi + ti;
        fftReal[start + offset + half] = br - tr; fftImag[start + offset + half] = bi - ti;
      }
    }
  }
}

float reflectedSample(int paddedIndex) {
  int index = paddedIndex - kCenterPadding;
  if (index < 0) index = -index;
  else if (index >= kModelSamples) index = 2 * kModelSamples - 2 - index;
  return modelSamples[index];
}

bool computeFeatures() {
  const uint32_t started = millis();
  float mean = 0.0f;
  for (int i = 0; i < kModelSamples; ++i) {
    modelSamples[i] = (static_cast<float>(recorderSamples[2 * i]) + recorderSamples[2 * i + 1]) /
                      (2.0f * 32768.0f);
    mean += modelSamples[i];
  }
  mean /= kModelSamples;
  float peak = 0.0f;
  for (int i = 0; i < kModelSamples; ++i) {
    modelSamples[i] -= mean;
    peak = max(peak, fabsf(modelSamples[i]));
  }
  if (peak < 1e-7f) return false;
  for (int i = 0; i < kModelSamples; ++i) modelSamples[i] /= peak;

  float sum = 0.0f;
  float pcenSmoother[kMels] = {};
  for (int frame = 0; frame < kFrames; ++frame) {
    const int frameStart = frame * 80;
    for (int sample = 0; sample < kFftSize; ++sample) {
      fftReal[sample] = reflectedSample(frameStart + sample) * hannWindow[sample];
      fftImag[sample] = 0.0f;
    }
    fft256();
    for (int mel = 0; mel < kMels; ++mel) {
      int left = kMelBins[mel], center = kMelBins[mel + 1], right = kMelBins[mel + 2];
      if (center <= left) center = min(left + 1, kFftBins - 1);
      if (right <= center) right = min(center + 1, kFftBins);
      float energy = 0.0f;
      // Match train_3class_cv.py::build_mel_filter exactly.  PyTorch's
      // linspace includes both endpoints within each half-open slice.  The
      // trained model therefore uses (count - 1) as the interpolation
      // denominator, with its defined one-sample behavior retained here.
      const int risingCount = center - left;
      for (int bin = left; bin < center; ++bin) {
        const float power = fftReal[bin] * fftReal[bin] + fftImag[bin] * fftImag[bin];
        const float weight = risingCount <= 1
            ? 0.0f
            : (bin - left) / static_cast<float>(risingCount - 1);
        energy += power * weight;
      }
      const int fallingCount = right - center;
      for (int bin = center; bin < right; ++bin) {
        const float power = fftReal[bin] * fftReal[bin] + fftImag[bin] * fftImag[bin];
        const float weight = fallingCount <= 1
            ? 1.0f
            : 1.0f - (bin - center) / static_cast<float>(fallingCount - 1);
        energy += power * weight;
      }
      energy = max(energy, 1e-8f);
      float value;
      if (usePcen) {
        if (frame == 0) pcenSmoother[mel] = energy;
        else pcenSmoother[mel] =
            (1.0f - kPcenSmoothing) * pcenSmoother[mel] + kPcenSmoothing * energy;
        const float normalized = energy /
            powf(kPcenEps + pcenSmoother[mel], kPcenAlpha);
        value = powf(normalized + kPcenDelta, kPcenR) - kPcenDeltaCompressed;
      } else {
        value = logf(energy);
      }
      logMelFeatures[mel * kFrames + frame] = value;
      sum += value;
    }
  }
  const float featureMean = sum / kFeatureCount;
  float squareSum = 0.0f;
  for (int i = 0; i < kFeatureCount; ++i) {
    const float difference = logMelFeatures[i] - featureMean;
    squareSum += difference * difference;
  }
  const float featureStd = sqrtf(squareSum / (kFeatureCount - 1));
  for (int i = 0; i < kFeatureCount; ++i) {
    logMelFeatures[i] = (logMelFeatures[i] - featureMean) / max(featureStd, 1e-5f);
  }
  lastFeatureMs = millis() - started;
  return true;
}

bool loadModel(int index) {
  if (loadedModel == index && interpreter) return true;
  if (interpreter) {
    interpreter->~MicroInterpreter();
    interpreter = nullptr;
  }
  memset(tensorArena, 0, kTensorArenaBytes);
  const tflite::Model* model = tflite::GetModel(kModels[index].data);
  if (!model || model->version() != TFLITE_SCHEMA_VERSION) return false;
  interpreter = new (interpreterStorage) tflite::MicroInterpreter(
      model, resolver, tensorArena, kTensorArenaBytes);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    interpreter->~MicroInterpreter(); interpreter = nullptr; return false;
  }
  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);
  if (!inputTensor || !outputTensor || inputTensor->type != kTfLiteInt8 ||
      outputTensor->type != kTfLiteInt8 || inputTensor->bytes != kFeatureCount ||
      outputTensor->bytes != kModels[index].classes) return false;
  loadedModel = index;
  return true;
}

void quantizeFeaturesForCurrentModel() {
  const float scale = inputTensor->params.scale;
  const int zero = inputTensor->params.zero_point;
  for (int i = 0; i < kFeatureCount; ++i) {
    const int value = static_cast<int>(roundf(logMelFeatures[i] / scale + zero));
    inputTensor->data.int8[i] = static_cast<int8_t>(constrain(value, -128, 127));
  }
}

bool invokeModel(int index, float* positiveScore, int* predictedClass, uint32_t* elapsedUs) {
  const uint32_t started = micros();
  if (!loadModel(index)) return false;
  quantizeFeaturesForCurrentModel();
  if (interpreter->Invoke() != kTfLiteOk) return false;
  *elapsedUs = micros() - started;
  int best = 0;
  for (int i = 1; i < kModels[index].classes; ++i) {
    if (outputTensor->data.int8[i] > outputTensor->data.int8[best]) best = i;
  }
  *predictedClass = best;
  if (kModels[index].classes == 2) {
    const float outScale = outputTensor->params.scale;
    const int outZero = outputTensor->params.zero_point;
    const float first = (outputTensor->data.int8[0] - outZero) * outScale;
    const float second = (outputTensor->data.int8[1] - outZero) * outScale;
    *positiveScore = 1.0f / (1.0f + expf(second - first));
  } else {
    *positiveScore = 0.0f;
  }
  return true;
}

void printRawOutput() {
  for (int i = 0; i < outputTensor->bytes; ++i) {
    if (i) Serial.print(',');
    Serial.print(outputTensor->data.int8[i]);
  }
}

void printStatus();

bool pollStopDuringWork() {
  while (Serial.available()) {
    const char command = static_cast<char>(Serial.read());
    if (command == 'E' || command == 'e') {
      stopLiveSession("user");
      return true;
    }
    if (command == 'A' || command == 'a' || command == 'B' || command == 'b') {
      Serial.println("FRONTEND_SELECT_ERROR stop_live_mode_first");
    }
    if (command == '?') printStatus();
  }
  return false;
}

bool scoreCurrentFeatures(int mode, bool useCalibration, bool allowLiveStop, int* predicted,
                          float* noMargin, uint32_t* totalInferenceUs) {
  *predicted = 0;
  *totalInferenceUs = 0;
  if (mode == 1) {
    for (int classIndex = 0; classIndex < 6; ++classIndex) {
      if (allowLiveStop && pollStopDuringWork()) return false;
      uint32_t elapsed = 0; int binaryClass = 0;
      if (!invokeModel(3 + classIndex, &lastClassScores[classIndex], &binaryClass, &elapsed)) return false;
      *totalInferenceUs += elapsed;
    }
    int bestMosquito = 0;
    for (int i = 1; i < 5; ++i) {
      if (lastClassScores[i] > lastClassScores[bestMosquito]) bestMosquito = i;
    }
    *noMargin = lastClassScores[5] - lastClassScores[bestMosquito];
    const float threshold = useCalibration ? calibrationThreshold[mode] : 0.0f;
    *predicted = *noMargin >= threshold ? 5 : bestMosquito;
  } else {
    const int modelIndex = mode == 2 ? 1 : (mode == 4 ? activeModelIndex() : 0);
    float unused = 0.0f; int rawPredicted = 0; uint32_t elapsed = 0;
    if (!invokeModel(modelIndex, &unused, &rawPredicted, &elapsed)) return false;
    *totalInferenceUs = elapsed;
    const float outputScale = outputTensor->params.scale;
    const int outputZero = outputTensor->params.zero_point;
    float maximumLogit = -1e30f;
    for (int i = 0; i < 6; ++i) {
      lastClassScores[i] = (outputTensor->data.int8[i] - outputZero) * outputScale;
      maximumLogit = max(maximumLogit, lastClassScores[i]);
    }
    float probabilitySum = 0.0f;
    for (int i = 0; i < 6; ++i) {
      lastClassScores[i] = expf(lastClassScores[i] - maximumLogit);
      probabilitySum += lastClassScores[i];
    }
    for (int i = 0; i < 6; ++i) lastClassScores[i] /= probabilitySum;
    int bestMosquito = 0;
    for (int i = 1; i < 5; ++i) {
      if (lastClassScores[i] > lastClassScores[bestMosquito]) bestMosquito = i;
    }
    *noMargin = lastClassScores[5] - lastClassScores[bestMosquito];
    const float threshold = useCalibration ? calibrationThreshold[mode] : 0.0f;
    *predicted = *noMargin >= threshold ? 5 : bestMosquito;
  }
  return true;
}

int classifyAveragedScores(int mode, float* noMargin) {
  int bestMosquito = 0;
  for (int index = 1; index < 5; ++index) {
    if (lastClassScores[index] > lastClassScores[bestMosquito]) bestMosquito = index;
  }
  *noMargin = lastClassScores[5] - lastClassScores[bestMosquito];
  const float threshold = liveUseCalibration ? calibrationThreshold[mode] : 0.0f;
  return *noMargin >= threshold ? 5 : bestMosquito;
}

// Diagnostic reference implementation of the exact PC aggregation geometry.
// It runs three CNN invocations for one fixed 3.0 s capture. The live mode below
// uses the same geometry with a sustainable, non-overlapping 3.0 s decision hop.
void runExact3sDiagnostic() {
  if (liveMode) {
    Serial.println("EXACT3S_ERROR stop_live_mode_first");
    return;
  }
  constexpr int mode = 4;  // overlap025 selected fold
  const uint64_t captureStart = capturedSampleCount();
  float averaged[6] = {};
  uint32_t inferenceTotalUs = 0;
  uint32_t featureTotalMs = 0;
  loadedModel = -1;
  Serial.println("EXACT3S_STARTED mode=4 context_centers_sec=0.75,1.50,2.25 capture_sec=3.0");
  for (int context = 0; context < 3; ++context) {
    const uint64_t endSample = captureStart + kExactContextEndOffsets[context];
    if (!snapshotContextAt(endSample, false)) {
      Serial.println("EXACT3S_ERROR capture_or_ring");
      return;
    }
    if (!computeFeatures()) {
      Serial.println("EXACT3S_ERROR feature");
      return;
    }
    featureTotalMs += lastFeatureMs;
    int predicted = 0; float noMargin = 0.0f; uint32_t inferenceUs = 0;
    if (!scoreCurrentFeatures(mode, false, false, &predicted, &noMargin, &inferenceUs)) {
      Serial.println("EXACT3S_ERROR inference");
      return;
    }
    inferenceTotalUs += inferenceUs;
    Serial.print("EXACT3S_CONTEXT index="); Serial.print(context + 1);
    Serial.print(" end_offset_samples="); Serial.print(kExactContextEndOffsets[context]);
    Serial.print(" scores=");
    for (int classIndex = 0; classIndex < 6; ++classIndex) {
      if (classIndex) Serial.print(',');
      Serial.print(lastClassScores[classIndex], 7);
      averaged[classIndex] += lastClassScores[classIndex] / 3.0f;
    }
    Serial.println();
  }
  for (int classIndex = 0; classIndex < 6; ++classIndex) lastClassScores[classIndex] = averaged[classIndex];
  float noMargin = 0.0f;
  const int predicted = classifyAveragedScores(mode, &noMargin);
  Serial.print("EXACT3S_RESULT class="); Serial.print(kClassNames[predicted]);
  Serial.print(" scores=");
  for (int classIndex = 0; classIndex < 6; ++classIndex) {
    if (classIndex) Serial.print(',');
    Serial.print(averaged[classIndex], 7);
  }
  Serial.print(" feature_total_ms="); Serial.print(featureTotalMs);
  Serial.print(" infer_total_ms="); Serial.print(inferenceTotalUs / 1000);
  Serial.println(" effective_audio_support_sec=3.0 decision_hop_sec=diagnostic_single_shot");
}

bool classifyExact3sLive(int mode, uint64_t decisionEndSample) {
  const uint32_t allStarted = millis();
  const uint64_t decisionStartSample = decisionEndSample - kExactDecisionRecorderSamples;
  float averaged[6] = {};
  uint32_t featureTotalMs = 0;
  uint32_t inferenceTotalUs = 0;
  for (int context = 0; context < 3; ++context) {
    const uint64_t contextEndSample = decisionStartSample + kExactContextEndOffsets[context];
    if (!snapshotContextAt(contextEndSample, true)) return false;
    if (pollStopDuringWork()) return false;
    if (!computeFeatures()) return false;
    featureTotalMs += lastFeatureMs;
    int contextPredicted = 0;
    float contextMargin = 0.0f;
    uint32_t inferenceUs = 0;
    if (!scoreCurrentFeatures(mode, false, true, &contextPredicted,
                              &contextMargin, &inferenceUs)) return false;
    inferenceTotalUs += inferenceUs;
    for (int classIndex = 0; classIndex < 6; ++classIndex) {
      averaged[classIndex] += lastClassScores[classIndex] / 3.0f;
    }
  }
  lastDecisionEndSample = decisionEndSample;
  lastFeatureMs = featureTotalMs;
  lastInferenceMs = inferenceTotalUs / 1000;
  for (int classIndex = 0; classIndex < 6; ++classIndex) {
    lastClassScores[classIndex] = averaged[classIndex];
  }
  float noMargin = 0.0f;
  const int predicted = classifyAveragedScores(mode, &noMargin);
  ++resultIndex;
  const uint64_t captured = capturedSampleCount();
  const uint32_t audioLagMs = captured >= decisionEndSample
      ? static_cast<uint32_t>((captured - decisionEndSample) * 1000ULL / kRecorderRate) : 0;
  const uint32_t latencyMs = millis() - allStarted;
  Serial.print("LIVE_RESULT index="); Serial.print(resultIndex);
  Serial.print(" mode=4 overlap025 exact_window_sec=3.0 decision_hop_sec=3.0");
  Serial.print(" context_centers_sec=0.75,1.50,2.25 class=");
  Serial.print(kClassNames[predicted]); Serial.print(" scores=");
  for (int i = 0; i < 6; ++i) {
    if (i) Serial.print(',');
    Serial.print(lastClassScores[i], 5);
  }
  Serial.print(" calibrated=0 no_margin="); Serial.print(noMargin, 5);
  Serial.print(" no_threshold=0.00000 feature_ms="); Serial.print(lastFeatureMs);
  Serial.print(" infer_ms="); Serial.print(lastInferenceMs);
  Serial.print(" audio_lag_ms="); Serial.print(audioLagMs);
  Serial.print(" latency_ms="); Serial.println(latencyMs);
  logPrediction(mode, predicted, noMargin, audioLagMs, latencyMs);
  return true;
}

void printStatus() {
  Serial.print("STATUS ready=1 live="); Serial.print(liveMode ? 1 : 0);
  Serial.print(" mode="); Serial.print(selectedMode);
  Serial.print(" live_calibrated="); Serial.print(liveUseCalibration ? 1 : 0);
  Serial.print(" context_ready="); Serial.print(contextReady ? 1 : 0);
  Serial.print(" results="); Serial.print(resultIndex);
  Serial.print(" last_capture_ms="); Serial.print(lastCaptureMs);
  Serial.print(" last_feature_ms="); Serial.print(lastFeatureMs);
  Serial.print(" last_infer_ms="); Serial.print(lastInferenceMs);
  Serial.print(" exact_window_sec=3 exact_hop_sec=3 context_centers_sec=0.75,1.50,2.25");
  Serial.print(" frontend="); Serial.print(activeFrontendName());
  Serial.print(" deployed_model="); Serial.print(activeModelName());
  Serial.print(" captured_samples="); Serial.print(static_cast<uint32_t>(capturedSampleCount()));
  Serial.print(" sd_ready="); Serial.print(sdReady ? 1 : 0);
  Serial.print(" recording="); Serial.print(recordingActive ? 1 : 0);
  Serial.print(" recording_error="); Serial.print(recordingError ? 1 : 0);
  Serial.print(" wav="); Serial.print(wavFilename[0] ? wavFilename : "none");
  Serial.print(" csv="); Serial.print(csvFilename[0] ? csvFilename : "none");
  Serial.print(" audio_seconds="); Serial.print(static_cast<float>(recordingSamplesWritten) / kRecorderRate, 3);
  Serial.print(" max_recording_seconds=3600");
  Serial.print(" dropped_blocks="); Serial.print(recordingDroppedBlocks);
  Serial.print(" skipped_hops="); Serial.print(skippedHopCount);
  Serial.print(" auto_start_pending="); Serial.print(autoStartPending ? 1 : 0);
  Serial.print(" i2s_slot="); Serial.print(activeI2sWordOffset);
  Serial.print(" i2s_slot_ready="); Serial.print(i2sSlotDetectionComplete ? 1 : 0);
  for (int mode = 1; mode <= 4; ++mode) {
    Serial.print(" cal"); Serial.print(mode); Serial.print('=');
    Serial.print(calibrationValid[mode] ? 1 : 0);
    Serial.print(':'); Serial.print(calibrationThreshold[mode], 5);
  }
  Serial.print(" free_heap="); Serial.print(ESP.getFreeHeap());
  Serial.print(" free_psram="); Serial.println(ESP.getFreePsram());
}

void startMode(int mode, bool useCalibration) {
  if (mode != 4) {
    Serial.println("LIVE_START_ERROR this_build_supports_mode_4_only");
    return;
  }
  if (liveMode || recordingActive || wavFile.isOpen() || csvFile.isOpen()) {
    Serial.println("LIVE_START_ERROR session_already_active");
    return;
  }
  if (useCalibration && !calibrationValid[mode]) {
    Serial.print("CALIBRATED_START_ERROR mode="); Serial.print(mode);
    Serial.print(" not_calibrated run_C"); Serial.print(mode); Serial.println("_first");
    return;
  }
  selectedMode = mode; liveMode = false; contextReady = false; resultIndex = 0;
  skippedHopCount = 0;
  lastDecisionEndSample = 0;
  liveUseCalibration = useCalibration;
  loadedModel = -1;
  while (Serial.available()) Serial.read();
  if (!startRecordingSession(mode, useCalibration)) {
    recordingActive = false;
    stopRecordingSession();
    selectedMode = 0;
    liveUseCalibration = false;
    Serial.println("LIVE_START_ERROR sd_session_initialization");
    return;
  }
  nextAudioEndSample = recordingFirstSample + kExactDecisionRecorderSamples;
  liveMode = true;
  Serial.print("LIVE_STARTED mode="); Serial.print(mode);
  Serial.print(" calibrated="); Serial.print(liveUseCalibration ? 1 : 0);
  Serial.print(" no_threshold="); Serial.print(liveUseCalibration ? calibrationThreshold[mode] : 0.0f, 5);
  Serial.print(" overlap025_selected_fold exact_window_sec=3 hop_sec=3 context_centers_sec=0.75,1.50,2.25 frontend=");
  Serial.println(activeFrontendName());
  Serial.print("SESSION_FILES wav="); Serial.print(wavFilename);
  Serial.print(" csv="); Serial.println(csvFilename);
}

void stopLiveSession(const char* reason) {
  const bool hadSession = liveMode || recordingActive || wavFile.isOpen() || csvFile.isOpen();
  liveMode = false;
  selectedMode = 0;
  liveUseCalibration = false;
  contextReady = false;
  awaitingCalibrationMode = false;
  pendingCalibrationMode = 0;
  if (hadSession) stopRecordingSession();
  Serial.print("LIVE_STOPPED reason="); Serial.println(reason);
}

void clearCalibration() {
  for (int mode = 1; mode <= 4; ++mode) {
    calibrationThreshold[mode] = 0.0f;
    calibrationValid[mode] = false;
  }
  Serial.println("CALIBRATION_RESET all_modes threshold=0.00000");
}

void runCalibration(int mode) {
  constexpr int kCalibrationSeconds = 10;
  constexpr int kCalibrationSamples = kRecorderRate * kCalibrationSeconds;
  constexpr int kCalibrationWindows = 12;
  Serial.print("CALIBRATION_STARTED mode="); Serial.print(mode);
  Serial.println(" recording=10s requirement=no_mosquito_no_playback");
  int16_t* calibrationPcm = static_cast<int16_t*>(heap_caps_malloc(
      kCalibrationSamples * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (!calibrationPcm) {
    Serial.println("CALIBRATION_ERROR BUFFER_ALLOCATION");
    return;
  }
  if (audioCaptureTaskHandle) vTaskSuspend(audioCaptureTaskHandle);
  i2s_zero_dma_buffer(kI2sPort);
  const bool calibrationCaptured = captureSamples(calibrationPcm, kCalibrationSamples);
  if (audioCaptureTaskHandle) vTaskResume(audioCaptureTaskHandle);
  if (!calibrationCaptured) {
    heap_caps_free(calibrationPcm);
    Serial.println("CALIBRATION_ERROR CAPTURE");
    return;
  }
  Serial.println("CALIBRATION_CAPTURE_DONE analyzing=12_windows");
  float margins[kCalibrationWindows];
  float calibrationScores[kCalibrationWindows][6] = {};
  const int maximumStart = kCalibrationSamples - kContextRecorderSamples;
  for (int window = 0; window < kCalibrationWindows; ++window) {
    const int start = (maximumStart * window) / (kCalibrationWindows - 1);
    memcpy(recorderSamples, calibrationPcm + start,
           kContextRecorderSamples * sizeof(int16_t));
    if (!computeFeatures()) {
      heap_caps_free(calibrationPcm);
      Serial.println("CALIBRATION_ERROR FEATURE");
      return;
    }
    int predicted = 0; uint32_t inferenceUs = 0;
    if (!scoreCurrentFeatures(mode, false, false, &predicted, &margins[window], &inferenceUs)) {
      heap_caps_free(calibrationPcm);
      Serial.println("CALIBRATION_ERROR INFERENCE");
      return;
    }
    for (int classIndex = 0; classIndex < 6; ++classIndex) {
      calibrationScores[window][classIndex] = lastClassScores[classIndex];
    }
    Serial.print("CALIBRATION_PROGRESS mode="); Serial.print(mode);
    Serial.print(" window="); Serial.print(window + 1); Serial.print('/');
    Serial.print(kCalibrationWindows); Serial.print(" no_margin=");
    Serial.print(margins[window], 5); Serial.print(" infer_ms=");
    Serial.println(inferenceUs / 1000);
  }
  heap_caps_free(calibrationPcm);
  int marginCount = kCalibrationWindows;
  if (mode != 1) {
    marginCount = kCalibrationWindows - kStandardScoreHistory + 1;
    for (int decision = 0; decision < marginCount; ++decision) {
      float averaged[6] = {};
      for (int history = 0; history < kStandardScoreHistory; ++history) {
        for (int classIndex = 0; classIndex < 6; ++classIndex) {
          averaged[classIndex] += calibrationScores[decision + history][classIndex]
                                  / kStandardScoreHistory;
        }
      }
      int bestMosquito = 0;
      for (int classIndex = 1; classIndex < 5; ++classIndex) {
        if (averaged[classIndex] > averaged[bestMosquito]) bestMosquito = classIndex;
      }
      margins[decision] = averaged[5] - averaged[bestMosquito];
    }
  }
  float ordered[kCalibrationWindows];
  memcpy(ordered, margins, marginCount * sizeof(float));
  for (int i = 0; i < marginCount - 1; ++i) {
    for (int j = i + 1; j < marginCount; ++j) {
      if (ordered[j] < ordered[i]) {
        const float value = ordered[i]; ordered[i] = ordered[j]; ordered[j] = value;
      }
    }
  }
  // The second-smallest margin is a robust approximation of the 10th
  // percentile.  The -0.02 guard targets roughly 90% background acceptance,
  // while the -0.25 floor prevents one bad calibration from granting an
  // unlimited No-mosquito veto.
  calibrationThreshold[mode] = constrain(ordered[1] - 0.02f, -0.25f, 0.0f);
  calibrationValid[mode] = true;
  contextReady = false;
  loadedModel = -1;
  Serial.print("CALIBRATION_DONE mode="); Serial.print(mode);
  Serial.print(" windows="); Serial.print(marginCount);
  Serial.print(" threshold="); Serial.print(calibrationThreshold[mode], 5);
  Serial.print(" margin_min="); Serial.print(ordered[0], 5);
  Serial.print(" margin_p10="); Serial.print(ordered[1], 5);
  const int middle = marginCount / 2;
  const float median = marginCount % 2 ? ordered[middle]
      : (ordered[middle - 1] + ordered[middle]) * 0.5f;
  Serial.print(" margin_median="); Serial.print(median, 5);
  Serial.print(" margin_max="); Serial.println(ordered[marginCount - 1], 5);
}

void runParityCapture() {
  contextReady = false;
  // The continuous capture task owns I2S.  Snapshot its ring without pausing
  // that task; suspending it while it holds an I2S read can deadlock the probe.
  const uint64_t parityEndSample = capturedSampleCount();
  const bool parityCaptured = snapshotContextAt(parityEndSample, false);
  if (!parityCaptured || !computeFeatures()) {
    Serial.println("PARITY_ERROR CAPTURE_OR_FEATURE"); return;
  }
  Serial.print("PARITY_BINARY_FOLLOWS bytes=");
  Serial.println(kContextRecorderSamples * sizeof(int16_t));
  if (!loadModel(activeModelIndex())) { Serial.println("PARITY_ERROR MODEL"); return; }
  quantizeFeaturesForCurrentModel();
  Serial.write(reinterpret_cast<const uint8_t*>("MMP1"), 4);
  sendUint32LE(kRecorderRate);
  sendUint32LE(kContextRecorderSamples);
  sendUint32LE(kFeatureCount);
  Serial.write(reinterpret_cast<const uint8_t*>(recorderSamples),
               kContextRecorderSamples * sizeof(int16_t));
  Serial.write(reinterpret_cast<const uint8_t*>(inputTensor->data.int8), kFeatureCount);
  Serial.flush();
  for (int modelIndex = activeModelIndex(); modelIndex <= activeModelIndex(); ++modelIndex) {
    float score = 0.0f; int predicted = 0; uint32_t elapsed = 0;
    if (!invokeModel(modelIndex, &score, &predicted, &elapsed)) {
      Serial.print("PARITY_MODEL_ERROR "); Serial.println(kModels[modelIndex].name); return;
    }
    Serial.print("PARITY_MODEL "); Serial.print(kModels[modelIndex].name);
    Serial.print(" raw="); printRawOutput();
    Serial.print(" us="); Serial.println(elapsed);
  }
  Serial.print("PARITY_DONE feature_ms="); Serial.println(lastFeatureMs);
}

// Host verification protocol.  V receives one already-quantized model tensor
// and exercises only the optimized TFLM graph.  This makes the complete 240
// item export-verification set reproducible without acoustic replay error.
void runHostFeatureVerification() {
  if (liveMode) { Serial.println("HOST_FEATURE_ERROR stop_live_mode_first"); return; }
  if (!loadModel(activeModelIndex())) { Serial.println("HOST_FEATURE_ERROR model"); return; }
  Serial.print("HOST_FEATURE_READY bytes="); Serial.println(kFeatureCount);
  Serial.flush();
  if (!readSerialExact(reinterpret_cast<uint8_t*>(inputTensor->data.int8),
                       kFeatureCount, 5000)) {
    Serial.println("HOST_FEATURE_ERROR timeout"); return;
  }
  const uint32_t started = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("HOST_FEATURE_ERROR invoke"); return;
  }
  const uint32_t elapsed = micros() - started;
  Serial.write(reinterpret_cast<const uint8_t*>("HFT1"), 4);
  sendUint32LE(elapsed);
  sendUint32LE(outputTensor->bytes);
  Serial.write(reinterpret_cast<const uint8_t*>(outputTensor->data.int8),
               outputTensor->bytes);
  Serial.flush();
}

// P receives one exact 1.5 s INMP441 PCM16 window, then returns both the board
// int8 active-frontend tensor and raw model output for PC comparison.
void runHostPcmParity() {
  if (liveMode) { Serial.println("HOST_PCM_ERROR stop_live_mode_first"); return; }
  Serial.print("HOST_PCM_READY bytes=");
  Serial.println(kContextRecorderSamples * sizeof(int16_t));
  Serial.flush();
  if (!readSerialExact(reinterpret_cast<uint8_t*>(recorderSamples),
                       kContextRecorderSamples * sizeof(int16_t), 5000)) {
    Serial.println("HOST_PCM_ERROR timeout"); return;
  }
  if (!computeFeatures()) { Serial.println("HOST_PCM_ERROR feature"); return; }
  if (!loadModel(activeModelIndex())) { Serial.println("HOST_PCM_ERROR model"); return; }
  quantizeFeaturesForCurrentModel();
  // MicroAllocator may reuse the input arena after its first consumer.  Keep
  // the pre-Invoke tensor in the now-unused PCM buffer for a valid export.
  memcpy(reinterpret_cast<uint8_t*>(recorderSamples),
         reinterpret_cast<const uint8_t*>(inputTensor->data.int8), kFeatureCount);
  const uint32_t inferenceStarted = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("HOST_PCM_ERROR invoke"); return;
  }
  const uint32_t elapsed = micros() - inferenceStarted;
  Serial.write(reinterpret_cast<const uint8_t*>("HPT1"), 4);
  sendUint32LE(kRecorderRate);
  sendUint32LE(kContextRecorderSamples);
  sendUint32LE(kFeatureCount);
  sendUint32LE(outputTensor->bytes);
  sendUint32LE(lastFeatureMs);
  sendUint32LE(elapsed);
  Serial.write(reinterpret_cast<const uint8_t*>(recorderSamples),
               kFeatureCount);
  Serial.write(reinterpret_cast<const uint8_t*>(outputTensor->data.int8),
               outputTensor->bytes);
  Serial.flush();
}

void setup() {
  Serial.begin(115200);
  const uint32_t deadline = millis() + 5000;
  while (!Serial && static_cast<int32_t>(millis() - deadline) < 0) delay(10);
  Serial.print("BOOT "); Serial.print(kFirmwareBuild);
  Serial.print(" frontend="); Serial.print(activeFrontendName());
  Serial.print(" deployed_model="); Serial.print(activeModelName());
  Serial.print(" speed_patch="); Serial.print(kSpeedPatchBuild);
  Serial.print(" pcen=0.6,2.0,0.25,0.8");
  Serial.println("; live inference + WAV + timestamped CSV");
  Serial.println("NOTICE published scores used five-fold ensembles; this firmware uses validation-selected folds");
  Serial.print("PSRAM bytes="); Serial.println(ESP.getPsramSize());
  if (!allocateBuffers()) { Serial.println("FATAL BUFFER_ALLOCATION"); return; }
  sdMutex = xSemaphoreCreateMutex();
  recordingQueue = xQueueCreate(kRecordingQueueBlocks, sizeof(RecordingBlock));
  if (!sdMutex || !recordingQueue) { Serial.println("FATAL RECORDING_QUEUE_ALLOCATION"); return; }
  sdReady = initialiseSDCard();
  if (!sdReady) { Serial.println("FATAL SD_CARD_OR_FILESYSTEM"); return; }
  if (xTaskCreatePinnedToCore(sdWriterTask, "sd_writer", 4096, nullptr, 1,
                              &sdWriterTaskHandle, 0) != pdPASS) {
    Serial.println("FATAL SD_WRITER_TASK"); return;
  }
  initialiseFft();
  microphoneReady = initialiseMicrophone();
  if (!microphoneReady) { Serial.println("FATAL MICROPHONE"); return; }
  if (!startAudioCaptureTask()) { Serial.println("FATAL CONTINUOUS_CAPTURE_TASK"); return; }
  if (resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddMaxPool2D() != kTfLiteOk ||
      resolver.AddMean() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk) {
    Serial.println("FATAL OP_RESOLVER"); return;
  }
  if (!loadModel(0)) { Serial.println("FATAL MODEL_ARENA"); return; }
  Serial.println("READY commands: A=select_logmel B=select_pcen 4=start_live_and_record E=stop_finalize ?=status N=list_sessions Jnnn=select_session D=download_selected_wav L=download_selected_csv X=live_parity V=host_feature_test P=host_pcm_parity");
  autoStartPending = true;
  autoStartDeadlineMs = millis() + kAutoStartDelayMs;
  Serial.print("AUTO_START_ARMED mode=4 delay_ms="); Serial.print(kAutoStartDelayMs);
  Serial.println(" cancel_with=E maximum_recording_seconds=3600");
}

void loop() {
  while (Serial.available()) {
    const char command = static_cast<char>(Serial.read());
    if (command != '\r' && command != '\n' && command != '?') {
      if (autoStartPending) Serial.println("AUTO_START_CANCELLED serial_command");
      autoStartPending = false;
    }
    if (awaitingSessionSelection) {
      if (command >= '0' && command <= '9' && sessionSelectionDigitCount < 3) {
        sessionSelectionDigits[sessionSelectionDigitCount++] = command;
        if (sessionSelectionDigitCount == 3) {
          sessionSelectionDigits[3] = '\0';
          awaitingSessionSelection = false;
          const uint16_t index = static_cast<uint16_t>(atoi(sessionSelectionDigits));
          if (selectSessionFiles(index)) {
            Serial.print("SESSION_SELECTED wav="); Serial.print(wavFilename);
            Serial.print(" csv="); Serial.println(csvFilename);
          } else {
            Serial.print("SESSION_SELECT_ERROR missing=LAB");
            Serial.println(sessionSelectionDigits);
          }
        }
      } else if (command != '\r' && command != '\n') {
        awaitingSessionSelection = false;
        sessionSelectionDigitCount = 0;
        Serial.println("SESSION_SELECT_ERROR expected_three_digits");
      }
      continue;
    }
    if (pendingCalibrationMode != 0) {
      const int mode = pendingCalibrationMode;
      pendingCalibrationMode = 0;
      if ((command == 'S' || command == 's') && !liveMode) {
        startMode(mode, true);
      } else if ((command == '\r' || command == '\n') && !liveMode) {
        runCalibration(mode);
      } else {
        Serial.println("CALIBRATION_COMMAND_ERROR use_C1_C2_C3_C4_or_C1S_C2S_C3S_C4S");
      }
      continue;
    }
    if (command == '\r' || command == '\n') continue;
    if (awaitingCalibrationMode) {
      awaitingCalibrationMode = false;
      if (command >= '1' && command <= '4' && !liveMode) {
        pendingCalibrationMode = command - '0';
        pendingCalibrationSinceMs = millis();
      } else {
        Serial.println("CALIBRATION_ERROR expected_mode_1_2_3_or_4");
      }
      continue;
    }
    if (command == 'A' || command == 'a' || command == 'B' || command == 'b') {
      if (liveMode || recordingActive) {
        Serial.println("FRONTEND_SELECT_ERROR stop_live_mode_first");
      } else {
        usePcen = command == 'B' || command == 'b';
        clearCalibration();
        Serial.print("FRONTEND_SELECTED frontend="); Serial.print(activeFrontendName());
        Serial.print(" deployed_model="); Serial.println(activeModelName());
      }
    }
    else if (command == '?') printStatus();
    else if (command == 'E' || command == 'e') {
      stopLiveSession("user");
    } else if (command >= '1' && command <= '4') startMode(command - '0', false);
    else if ((command == 'C' || command == 'c') && !liveMode) {
      awaitingCalibrationMode = true;
    }
    else if ((command == 'R' || command == 'r') && !liveMode) clearCalibration();
    else if ((command == 'N' || command == 'n') && !liveMode) printSessionList();
    else if ((command == 'J' || command == 'j') && !liveMode) {
      awaitingSessionSelection = true;
      sessionSelectionDigitCount = 0;
      memset(sessionSelectionDigits, 0, sizeof(sessionSelectionDigits));
      Serial.println("SESSION_SELECT_READY digits=3");
    }
    else if ((command == 'D' || command == 'd') && !liveMode) sendLatestSessionFile(false);
    else if ((command == 'L' || command == 'l') && !liveMode) sendLatestSessionFile(true);
    else if ((command == 'G' || command == 'g') && !liveMode) runExact3sDiagnostic();
    else if ((command == 'X' || command == 'x') && !liveMode) runParityCapture();
    else if ((command == 'V' || command == 'v') && !liveMode) runHostFeatureVerification();
    else if ((command == 'P' || command == 'p') && !liveMode) runHostPcmParity();
  }
  if (pendingCalibrationMode != 0 && millis() - pendingCalibrationSinceMs >= 150) {
    const int mode = pendingCalibrationMode;
    pendingCalibrationMode = 0;
    if (!liveMode) runCalibration(mode);
  }
  if (autoStartPending && static_cast<int32_t>(millis() - autoStartDeadlineMs) >= 0) {
    autoStartPending = false;
    Serial.println("AUTO_START_TRIGGER mode=4");
    startMode(4, false);
  }
  if (recordingError && (liveMode || recordingActive)) {
    stopLiveSession("recording_error");
    return;
  }
  if (liveMode && recordingFirstSample != UINT64_MAX &&
      capturedSampleCount() >= recordingFirstSample + kMaximumRecordingSamples) {
    stopLiveSession("maximum_recording_duration_3600s");
    return;
  }
  if (!liveMode) { delay(5); return; }
  currentResultStartMs = millis();
  if (!classifyExact3sLive(selectedMode, nextAudioEndSample)) {
    if (liveMode) {
      Serial.println("LIVE_ERROR EXACT3S_CAPTURE_FEATURE_OR_INFERENCE");
      stopLiveSession("exact3s_error");
    }
    return;
  }
  nextAudioEndSample += kExactDecisionRecorderSamples;
}
