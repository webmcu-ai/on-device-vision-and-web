// =============================================================================
//  XIAO ESP32S3 SENSE / XIAO ML KIT  -  FULL VISION ML  -  firmware-v007
//
//  Collect small images, train a tiny CNN, and classify what the camera sees,
//  all on the microcontroller. A companion web page can do the training faster.
//
//  By Jeremy Ellis   (MIT license, use at your own risk)
//  With free tier assistance from Claude (code overview), ChatGPT (critique),
//  Gemini (research) and Copilot (alternate).
//  GitHub   https://github.com/hpssjellis
//  LinkedIn https://www.linkedin.com/in/jeremy-ellis-4237a9bb/
//
// -----------------------------------------------------------------------------
//  THE MENU  (Serial monitor at 115200 baud and the 72x40 OLED)
//    1..NUM_CLASSES  collect images for that class (one folder per class)
//    NUM_CLASSES+1   Train  (on the device)
//    NUM_CLASSES+2   Infer  (live camera classification)
//
//  CONTROLS
//    Touch pad on pin A0:   1-2 taps = next item / capture / train again
//                           3+ taps  = select / exit back to the menu
//    Serial monitor keys:   t = next / capture / train again
//                           l = select / exit
//                           1..9 = jump straight to a menu item (menu only)
//
// -----------------------------------------------------------------------------
//  THE SD CARD
//    /images/<class name>/img_*.jpg   240x240 photos, one folder per class
//    /header/myWeights.bin            trained weights (the device loads this at boot)
//    /header/myWeights.h              same weights as C arrays (for baking into the sketch)
//    /header/config.json              class names + layout, written by the web page
//
// -----------------------------------------------------------------------------
//  TWO WAYS TO USE IT
//   A) Train on the device:  collect images -> Train -> Infer.
//   B) Train on the web page: collect images, move the SD card to the computer,
//      open the Vision CNN SD trainer page, train, save, put the card back,
//      reboot the device, choose Infer.
//
//  Way B can use a BIGGER model than the device could train (for example 96x96
//  input) because inference needs far less memory than training. Train will then
//  say "Not enough memory" but Infer still works.
//
// -----------------------------------------------------------------------------
//  SETTINGS THAT MUST MATCH THE WEB PAGE  (they are fixed when the sketch compiles)
//    INPUT_SIZE, CONV1_FILTERS, CONV2_FILTERS, NUM_CLASSES and myClassLabels[]
//  The page shows the exact lines to paste (section 2 "Classes and data").
//  If they do not match, the device refuses myWeights.bin at boot and prints
//  "REFUSED myWeights.bin" with the sizes it expected. Fix the settings, reflash.
//  Changing a class name only (same count) needs no reflash: the page writes
//  config.json and the device reads the names from it at boot.
//
//  WEIGHTS PRIORITY:  SD card myWeights.bin  >  baked-in myWeights.h  >  random start
//  To bake weights in: copy /header/myWeights.h next to this sketch, set the settings
//  above to match, then remove the // in front of  #define USE_BAKED_WEIGHTS  below.
//
// -----------------------------------------------------------------------------
//  WEB PAGE DEBUG FRAMES
//  Nothing extra is printed unless the web page sends 'D' (it repeats it every
//  5 s; the device stops after 15 s of silence, or when it receives 'd').
//  While on, the device prints "@F ..." lines holding the camera JPEG (base64),
//  the class probabilities and a heatmap, so the page can compare its own model
//  with the device on exactly the same picture.
//
// -----------------------------------------------------------------------------
//  ARDUINO IDE SETUP
//    Board: XIAO_ESP32S3     PSRAM: OPI PSRAM     Library: U8g2 (olikraus)
//  PLATFORMIO (platformio.ini)
//    lib_deps = olikraus/U8g2 @ ^2.35.30
//    build_flags = -DBOARD_HAS_PSRAM -DARDUINO_USB_CDC_ON_BOOT=1
//    board_build.arduino.memory_type = qio_opi
//    board_build.flash_mode = qio
//    board_upload.flash_size = 8MB
// =============================================================================


// Remove the // below AFTER copying myWeights.h from the SD card into the sketch folder.
//#define USE_BAKED_WEIGHTS

#ifdef USE_BAKED_WEIGHTS
  #include "myWeights.h"
#endif

#include "esp_camera.h"
#include "img_converters.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <algorithm>
#include <U8g2lib.h>
#include <Wire.h>
#include "mbedtls/base64.h"


// =============================================================================
//  SETTINGS YOU MAY CHANGE
// =============================================================================

// ---- Classes -----------------------------------------------------------------
// Number of things to recognise. The label text is also the SD folder name.
// Keep a number in front so the order stays the same. The menu has 9 digit keys,
// so up to 7 classes can be picked by digit; more still work with t and l.
#define NUM_CLASSES 3
String myClassLabels[NUM_CLASSES] = {"0Blank", "1Cup", "2Pen"};

// ---- Model layout (must match the web page, see the header comment) -----------
// INPUT_SIZE: square and EVEN. Photos on the SD card are always 240x240 and are
// shrunk to INPUT_SIZE when loaded. Cost grows with INPUT_SIZE squared.
// Training on the device needs much more memory than inference.
#define INPUT_SIZE    64
#define CONV1_FILTERS 4
#define CONV2_FILTERS 8

// ---- Training settings (on-device training only) ------------------------------
float LEARNING_RATE    = 0.0003;
int   BATCH_SIZE       = 6;
int   TARGET_EPOCHS    = 20;
int   VALIDATION_IMAGES = 3;    // last N images per class are held out to measure accuracy (0 = off)

// ---- Touch pad (A0) -----------------------------------------------------------
// Readings above myThresholdPress count as a touch; below myThresholdRelease as released.
const int myThresholdPress   = 1100;
const int myThresholdRelease = 900;

// ---- Camera picture settings --------------------------------------------------
// The web page shows images upright and mirrored left-right. To match it, the sensor is
// mirrored AND flipped vertically. Brightness and AE level go from -2 to 2 (0 = default).
// If pictures look darker than on the web page raise them; if they wash out, lower them.
// Photos already on the SD card are not changed by these settings.
#define MY_CAM_HMIRROR        1
#define MY_CAM_VFLIP          1
#define MY_CAM_BRIGHTNESS     1
#define MY_CAM_AE_LEVEL       1
#define MY_CAM_WARMUP_FRAMES  5     // frames thrown away at start so auto exposure can settle


// =============================================================================
//  DERIVED SIZES  (calculated from the settings above, do not edit)
// =============================================================================
// The convolutions are 3x3 and fixed in the code below.
#define CONV1_IN_STRIDE  27                                   // weights per conv1 filter: 3x3x3 (RGB)
#define CONV2_IN_STRIDE  (CONV1_FILTERS * 9)                  // weights per conv2 filter: 3x3xCONV1_FILTERS

#define CONV1_WEIGHTS    (CONV1_IN_STRIDE * CONV1_FILTERS)
#define CONV2_WEIGHTS    (CONV2_IN_STRIDE * CONV2_FILTERS)

#define CONV1_OUTPUT_SIZE (INPUT_SIZE - 2)
#define POOL1_OUTPUT_SIZE (CONV1_OUTPUT_SIZE / 2)
#define CONV2_OUTPUT_SIZE (POOL1_OUTPUT_SIZE - 2)
#define FLATTENED_SIZE    (CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * CONV2_FILTERS)

#define CONV1_OUT_FLOATS  (CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE * CONV1_FILTERS)
#define POOL1_OUT_FLOATS  (POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE * CONV1_FILTERS)

#define OUTPUT_WEIGHTS   (FLATTENED_SIZE * NUM_CLASSES)

// Exact size in bytes of /header/myWeights.bin for this layout.
#define MY_EXPECTED_WEIGHT_BYTES ((size_t)(CONV1_WEIGHTS + CONV1_FILTERS + CONV2_WEIGHTS + CONV2_FILTERS + OUTPUT_WEIGHTS + NUM_CLASSES) * 4)

static_assert(INPUT_SIZE % 2 == 0, "INPUT_SIZE must be even");
static_assert(CONV2_OUTPUT_SIZE >= 1, "INPUT_SIZE is too small");
static_assert(NUM_CLASSES >= 2, "Need at least 2 classes");

// If baked weights are used, the compiler checks that myWeights.h has the same layout.
#ifdef USE_BAKED_WEIGHTS
static_assert(sizeof(myModel_conv1_w)  / sizeof(float) == CONV1_WEIGHTS,  "myWeights.h conv1 size differs from this sketch: match INPUT_SIZE / filters / NUM_CLASSES");
static_assert(sizeof(myModel_conv1_b)  / sizeof(float) == CONV1_FILTERS,  "myWeights.h conv1 bias size differs from this sketch");
static_assert(sizeof(myModel_conv2_w)  / sizeof(float) == CONV2_WEIGHTS,  "myWeights.h conv2 size differs from this sketch: match INPUT_SIZE / filters / NUM_CLASSES");
static_assert(sizeof(myModel_conv2_b)  / sizeof(float) == CONV2_FILTERS,  "myWeights.h conv2 bias size differs from this sketch");
static_assert(sizeof(myModel_output_w) / sizeof(float) == OUTPUT_WEIGHTS, "myWeights.h output size differs from this sketch: match INPUT_SIZE / filters / NUM_CLASSES");
static_assert(sizeof(myModel_output_b) / sizeof(float) == NUM_CLASSES,    "myWeights.h output bias size differs from this sketch");
#endif


// =============================================================================
//  XIAO ESP32S3 CAMERA PINS
// =============================================================================
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13


// =============================================================================
//  GLOBAL VARIABLES  (all declared here, above every function)
// =============================================================================

// ---- Display ------------------------------------------------------------------
U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

// ---- Hardware status ----------------------------------------------------------
bool mySDavailable = false;       // true when the SD card mounted
bool myCameraOk    = false;       // true when the camera started
bool myWeightsTrained = false;    // true once usable weights are in memory

// ---- Menu ---------------------------------------------------------------------
const int myTotalItems = NUM_CLASSES + 2;   // one per class, then Train, then Infer
int  myMenuIndex = 1;
bool myIsSelected = false;
unsigned long myLastActivityTime = 0;
unsigned long myLastTapTime = 0;
const int myTapCooldown = 250;

// ---- Touch input: counts taps inside a short time window ----------------------
struct TouchState {
  bool isTouching = false;
  int tapCount = 0;
  unsigned long firstTapTime = 0;
  unsigned long lastReleaseTime = 0;
  unsigned long lastCheckTime = 0;
};
TouchState myTouch;
const unsigned long myTapWindow = 800;     // ms to wait for more taps
const int           myLongPressTaps = 3;   // this many taps counts as a "long press" (select / exit)
const unsigned long myDebounceMs = 50;

// ---- Image buffers ------------------------------------------------------------
uint8_t* myRgbBuffer = nullptr;            // one decoded 240x240 RGB picture, reused everywhere
int myResizeLut[INPUT_SIZE];               // source pixel for each model pixel (nearest neighbour)

// ---- Model weights (PSRAM) ----------------------------------------------------
float* myInputBuffer = nullptr;
float* myConv1_w = nullptr;
float* myConv1_b = nullptr;
float* myConv2_w = nullptr;
float* myConv2_b = nullptr;
float* myOutput_w = nullptr;
float* myOutput_b = nullptr;

// ---- Forward pass buffers (needed for inference) ------------------------------
float* myConv1_output = nullptr;
float* myPool1_output = nullptr;
float* myConv2_output = nullptr;
float* myDense_output = nullptr;           // class probabilities after softmax

// ---- Training-only buffers: allocated the first time Train is opened ----------
bool myTrainingBuffersReady = false;
float* myConv1_w_grad = nullptr;
float* myConv1_b_grad = nullptr;
float* myConv2_w_grad = nullptr;
float* myConv2_b_grad = nullptr;
float* myOutput_w_grad = nullptr;
float* myOutput_b_grad = nullptr;
float* myConv1_w_m = nullptr;              // Adam optimizer memory (m and v)
float* myConv1_w_v = nullptr;
float* myConv1_b_m = nullptr;
float* myConv1_b_v = nullptr;
float* myConv2_w_m = nullptr;
float* myConv2_w_v = nullptr;
float* myConv2_b_m = nullptr;
float* myConv2_b_v = nullptr;
float* myOutput_w_m = nullptr;
float* myOutput_w_v = nullptr;
float* myOutput_b_m = nullptr;
float* myOutput_b_v = nullptr;
float* myDense_grad = nullptr;             // back-propagation work buffers
float* myConv2_grad = nullptr;
float* myPool1_grad = nullptr;
float* myConv1_grad = nullptr;

// One entry per training buffer, used by myAllocateTrainingBuffers().
struct MyBufSpec {
  float** ptr;
  size_t count;
  bool zero;
};

// ---- Training data list -------------------------------------------------------
struct TrainingItem {
  String path;
  int label;
};
std::vector<TrainingItem> myTrainingData;

// ---- Web page debug frames ----------------------------------------------------
bool myDebugStream = false;
unsigned long myDebugLastSeen = 0;


// =============================================================================
//  FORWARD DECLARATIONS  (every function, so the order below does not matter)
// =============================================================================

// math helpers
inline float clip_value(float v, float mn = -100, float mx = 100);
inline float leaky_relu(float x);
inline float leaky_relu_deriv(float x);

// OLED
void myShowMessage(const char* l1, const char* l2 = nullptr, const char* l3 = nullptr);
void myRenderRgbToOLED(int imageCount);
void myDisplayImageOnOLED(camera_fb_t* fb, int imageCount);

// touch
int  myReadTouch();
void myResetTouchState();
void myUpdateTouchState();
int  myCheckTouchInput();
void myCheckTouchBackground();
int  myPeekTouchAction();

// memory and weights
void   myAllocateMemory();
size_t myTrainingBytesNeeded();
bool   myAllocateTrainingBuffers();
void   myResetAdam();
static void myDumpArray(File& file, const char* name, const float* data, int size);
void   myExportHeader();
bool   myLoadWeights();
void   mySaveWeights();

// config.json
static bool myJsonInt(const String& t, const char* key, int& out);
static int  myJsonStringArray(const String& t, const char* key, std::vector<String>& out);
void myLoadConfig();

// debug frames
bool myHandleDebugChar(char c);
static void myPrintB64(const uint8_t* d, size_t n);
void myDebugSendFrame(char kind, int n, camera_fb_t* fb, int pred, const float* logits);

// images
void   myResizeToInput(const uint8_t* rgb240, float* dst);
bool   myLoadImageFromFile(const char* path, float* buf);
String myUniqueImagePath(const String& folder);

// the CNN
void myForwardPass(float* input, float* logits);
void myBackwardDense(int label);
void myBackwardConv2();
void myBackwardPool1();
void myBackwardConv1();
void myAdamUpdate(float* w, float* g, float* m, float* v, int size, int step);
void myUpdateWeights(int step);

// menu actions
void myActionCollect(int classIdx);
bool myTrainingExitRequested();
void myActionTrain();
void myActionInfer();

// menu
String myMenuLabel(int item);
void myResetMenuState();
void myDrawMenu();
void myExecuteMenuItem(int idx);
void myHandleMenuNavigation();


// =============================================================================
//  SETUP AND LOOP
// =============================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  delay(1000);

  Serial.println("\n=== XIAO ESP32S3 Vision ML (firmware-v007) ===");
  Serial.printf("Layout: INPUT_SIZE %d, CONV1_FILTERS %d, CONV2_FILTERS %d, NUM_CLASSES %d\n",
                INPUT_SIZE, CONV1_FILTERS, CONV2_FILTERS, NUM_CLASSES);
  Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());

  myRgbBuffer = (uint8_t*)ps_malloc(240 * 240 * 3);
  if (!myRgbBuffer) {
    Serial.println("FATAL: could not allocate the picture buffer. Is PSRAM set to OPI PSRAM?");
    myShowMessage("PSRAM ERROR!", "Set PSRAM to", "OPI PSRAM");
    while (1) { delay(1000); }
  }

  // Pre-compute which source pixel feeds each model pixel (240x240 -> INPUT_SIZE).
  for (int i = 0; i < INPUT_SIZE; i++) {
    myResizeLut[i] = min((int)((i + 0.5) * 240.0 / INPUT_SIZE), 239);
  }

  pinMode(A0, INPUT);
  u8g2.begin();
  u8g2.setFont(u8g2_font_6x10_tf);

  // ---- SD card (started slowly with a timeout so a missing card cannot hang the boot) ----
  pinMode(21, OUTPUT);
  digitalWrite(21, HIGH);
  delay(100);
  Serial.println("Checking SD card...");
  SPI.begin();
  SPI.setFrequency(400000);
  mySDavailable = SD.begin(21, SPI, 400000, "/sd", 5, false);

  if (!mySDavailable) {
    SD.end();
    Serial.println("No SD card - continuing without it");
    myShowMessage("No SD card");
    delay(2000);
  } else {
    Serial.println("SD card mounted successfully");
    myLoadConfig();      // class names from /header/config.json
  }

  // ---- Camera ----
  camera_config_t config = {};     // all fields start at zero (= library defaults)
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM; config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_240X240;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  esp_err_t camErr = esp_camera_init(&config);
  if (camErr != ESP_OK) {
    Serial.printf("Camera init FAILED: 0x%x\n", camErr);
  } else {
    myCameraOk = true;
    Serial.println("Camera initialized");
    sensor_t* s = esp_camera_sensor_get();
    if (s != NULL) {
      s->set_hmirror(s, MY_CAM_HMIRROR);
      s->set_vflip(s, MY_CAM_VFLIP);
      s->set_brightness(s, MY_CAM_BRIGHTNESS);
      s->set_ae_level(s, MY_CAM_AE_LEVEL);
    }
    // Throw away the first frames so auto exposure settles before any image is used.
    for (int i = 0; i < MY_CAM_WARMUP_FRAMES; i++) {
      camera_fb_t* warm = esp_camera_fb_get();
      if (warm) esp_camera_fb_return(warm);
      delay(60);
    }
  }

  // Quieter serial output: only warnings and errors from the ESP-IDF libraries.
  esp_log_level_set("*", ESP_LOG_WARN);
  esp_log_level_set("esp_camera", ESP_LOG_ERROR);

  myAllocateMemory();      // inference memory + random starting weights

  Serial.printf("Weight file for this layout: %u bytes\n", (unsigned)MY_EXPECTED_WEIGHT_BYTES);
  Serial.printf("Training on the device needs about %u KB of PSRAM (free now: %u KB)\n",
                (unsigned)(myTrainingBytesNeeded() / 1024), (unsigned)(ESP.getFreePsram() / 1024));

#ifdef USE_BAKED_WEIGHTS
  memcpy(myConv1_w,  myModel_conv1_w,  CONV1_WEIGHTS  * sizeof(float));
  memcpy(myConv1_b,  myModel_conv1_b,  CONV1_FILTERS  * sizeof(float));
  memcpy(myConv2_w,  myModel_conv2_w,  CONV2_WEIGHTS  * sizeof(float));
  memcpy(myConv2_b,  myModel_conv2_b,  CONV2_FILTERS  * sizeof(float));
  memcpy(myOutput_w, myModel_output_w, OUTPUT_WEIGHTS * sizeof(float));
  memcpy(myOutput_b, myModel_output_b, NUM_CLASSES    * sizeof(float));
  Serial.println("Baked-in weights loaded from myWeights.h");
  myWeightsTrained = true;
#endif

  if (myLoadWeights()) {
    Serial.println("SD weights loaded - overriding baked-in weights");
  }

  myLastActivityTime = millis();
  myResetMenuState();
  delay(2000);             // time to open the serial monitor

  Serial.println("System ready - Tap A0 to navigate, 3+ taps to select");
  myDrawMenu();
}

void loop() {
  myHandleMenuNavigation();
}


// =============================================================================
//  MATH HELPERS
// =============================================================================
inline float clip_value(float v, float mn, float mx) {
  if (isnan(v) || isinf(v)) return 0;
  return constrain(v, mn, mx);
}

inline float leaky_relu(float x) { return x > 0 ? x : 0.1f * x; }
inline float leaky_relu_deriv(float x) { return x > 0 ? 1.0f : 0.1f; }


// =============================================================================
//  OLED HELPERS
// =============================================================================

// Up to three short lines (about 14 characters each fit on the 72x40 display).
void myShowMessage(const char* l1, const char* l2, const char* l3) {
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x7_tf);
    if (l1) u8g2.drawStr(0, 10, l1);
    if (l2) u8g2.drawStr(0, 22, l2);
    if (l3) u8g2.drawStr(0, 34, l3);
  } while (u8g2.nextPage());
}

// Draws myRgbBuffer (must already hold a 240x240 picture) as a black and white image.
// imageCount >= 0 shows that number in the corner (after a capture); -1 shows "LIVE".
void myRenderRgbToOLED(int imageCount) {
  int myOledWidth  = u8g2.getDisplayWidth();
  int myOledHeight = u8g2.getDisplayHeight();
  int myScaleX = 240 / myOledWidth;
  int myScaleY = 240 / myOledHeight;

  u8g2.firstPage();
  do {
    for (int myOledX = 0; myOledX < myOledWidth; myOledX++) {
      for (int myOledY = 0; myOledY < myOledHeight; myOledY++) {
        size_t myPixelIndex = ((myOledY * myScaleY) * 240 + (myOledX * myScaleX)) * 3;
        uint8_t myBrightness = (myRgbBuffer[myPixelIndex]     +
                                myRgbBuffer[myPixelIndex + 1] +
                                myRgbBuffer[myPixelIndex + 2]) / 3;
        if (myBrightness > 100) u8g2.drawPixel(myOledX, myOledY);
      }
    }
    if (imageCount >= 0) {
      u8g2.setFont(u8g2_font_ncenB10_tr);
      u8g2.setColorIndex(0);
      u8g2.drawBox(0, 0, 20, 15);
      u8g2.setColorIndex(1);
      u8g2.setCursor(3, 10);
      u8g2.print(String(imageCount));
    } else {
      u8g2.setFont(u8g2_font_5x7_tf);
      u8g2.setColorIndex(0);
      u8g2.drawBox(50, 0, 22, 8);
      u8g2.setColorIndex(1);
      u8g2.drawStr(52, 7, "LIVE");
    }
  } while (u8g2.nextPage());
}

// Decode a camera JPEG into myRgbBuffer and show it with a count badge.
void myDisplayImageOnOLED(camera_fb_t* fb, int imageCount) {
  if (!myRgbBuffer) return;
  if (!fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) {
    Serial.println("Failed to convert JPEG to RGB888 for OLED");
    return;
  }
  myRenderRgbToOLED(imageCount);
}


// =============================================================================
//  TOUCH INPUT
//  A tap is a touch and release. Taps inside myTapWindow are counted together:
//    1-2 taps -> "tap"  (returns 1)      3 or more taps -> "long press" (returns 2)
//  Call myCheckTouchInput() from a loop to receive the result once the window ends.
// =============================================================================

// Average of three readings of the touch pad.
int myReadTouch() {
  int sum = 0;
  for (int i = 0; i < 3; i++) {
    sum += analogRead(A0);
    delayMicroseconds(100);
  }
  return sum / 3;
}

void myResetTouchState() {
  myTouch.isTouching = false;
  myTouch.tapCount = 0;
  myTouch.firstTapTime = 0;
  myTouch.lastReleaseTime = 0;
  myTouch.lastCheckTime = 0;
}

// Reads the pad (at most every 20 ms) and counts taps.
void myUpdateTouchState() {
  unsigned long now = millis();
  if (now - myTouch.lastCheckTime < 20) return;
  myTouch.lastCheckTime = now;

  int val = myReadTouch();
  bool touchActive = myTouch.isTouching ? (val > myThresholdRelease) : (val > myThresholdPress);

  // A new touch just started.
  if (touchActive && !myTouch.isTouching) {
    if (now - myTouch.lastReleaseTime < myDebounceMs) return;
    myTouch.isTouching = true;

    if (myTouch.tapCount == 0 || (now - myTouch.firstTapTime < myTapWindow)) {
      if (myTouch.tapCount == 0) myTouch.firstTapTime = now;
      myTouch.tapCount++;
      Serial.printf("Tap #%d\n", myTouch.tapCount);
    } else {
      myTouch.tapCount = 1;               // the window ran out: start counting again
      myTouch.firstTapTime = now;
      Serial.println("Tap #1 (new window)");
    }
  }

  // The touch ended.
  if (!touchActive && myTouch.isTouching) {
    myTouch.isTouching = false;
    myTouch.lastReleaseTime = now;
  }
}

// Returns 0 = nothing yet, 1 = tap, 2 = long press (3+ taps). Consumes the result.
int myCheckTouchInput() {
  myUpdateTouchState();
  unsigned long now = millis();

  if (myTouch.tapCount > 0 && !myTouch.isTouching) {
    if (now - myTouch.firstTapTime > myTapWindow) {
      int result = (myTouch.tapCount >= myLongPressTaps) ? 2 : 1;
      int count = myTouch.tapCount;
      myResetTouchState();
      if (result == 2) Serial.printf("LONG PRESS detected (%d taps)\n", count);
      else             Serial.printf("TAP detected (%d tap%s)\n", count, count > 1 ? "s" : "");
      return result;
    }
  }
  return 0;
}

// Keeps the tap counter up to date without consuming a result (use inside heavy loops).
void myCheckTouchBackground() {
  myUpdateTouchState();
}

// Same answer as myCheckTouchInput() but does not consume it.
int myPeekTouchAction() {
  myUpdateTouchState();
  unsigned long now = millis();
  if (myTouch.tapCount > 0 && !myTouch.isTouching) {
    if (now - myTouch.firstTapTime > myTapWindow) {
      return (myTouch.tapCount >= myLongPressTaps) ? 2 : 1;
    }
  }
  return 0;
}


// =============================================================================
//  MEMORY
//  Inference needs the weights and the forward buffers (allocated at boot).
//  Training also needs gradients, Adam memory and back-propagation buffers.
//  Those are allocated only when Train is opened, so a model too big to train on
//  the device can still run inference.
// =============================================================================
void myAllocateMemory() {
  if (myInputBuffer != nullptr) return;

  Serial.println("\n=== Allocating inference memory ===");

  myInputBuffer = (float*)ps_malloc(INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));
  myConv1_w  = (float*)ps_malloc(CONV1_WEIGHTS * sizeof(float));
  myConv1_b  = (float*)ps_malloc(CONV1_FILTERS * sizeof(float));
  myConv2_w  = (float*)ps_malloc(CONV2_WEIGHTS * sizeof(float));
  myConv2_b  = (float*)ps_malloc(CONV2_FILTERS * sizeof(float));
  myOutput_w = (float*)ps_malloc(OUTPUT_WEIGHTS * sizeof(float));
  myOutput_b = (float*)ps_malloc(NUM_CLASSES * sizeof(float));

  myConv1_output = (float*)ps_malloc(CONV1_OUT_FLOATS * sizeof(float));
  myPool1_output = (float*)ps_malloc(POOL1_OUT_FLOATS * sizeof(float));
  myConv2_output = (float*)ps_malloc(FLATTENED_SIZE * sizeof(float));
  myDense_output = (float*)ps_malloc(NUM_CLASSES * sizeof(float));

  if (!myInputBuffer || !myConv1_w || !myConv1_b || !myConv2_w || !myConv2_b ||
      !myOutput_w || !myOutput_b || !myConv1_output || !myPool1_output ||
      !myConv2_output || !myDense_output) {
    Serial.println("FATAL: PSRAM allocation failed! Even inference does not fit.");
    Serial.println("Use a smaller INPUT_SIZE, or check that PSRAM is set to OPI PSRAM.");
    myShowMessage("PSRAM ERROR!", "Model too big", "or no PSRAM");
    while (1) { delay(1000); }
  }

  Serial.printf("Free PSRAM after inference allocation: %d bytes\n", ESP.getFreePsram());

  // Random starting weights (He initialization).
  float c1std = sqrt(2.0 / (9.0 * 3));
  for (int i = 0; i < CONV1_WEIGHTS; i++) myConv1_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * c1std;
  for (int i = 0; i < CONV1_FILTERS; i++) myConv1_b[i] = 0;

  float c2std = sqrt(2.0 / (double)CONV2_IN_STRIDE);
  for (int i = 0; i < CONV2_WEIGHTS; i++) myConv2_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * c2std;
  for (int i = 0; i < CONV2_FILTERS; i++) myConv2_b[i] = 0;

  float dstd = sqrt(2.0 / FLATTENED_SIZE);
  for (int i = 0; i < OUTPUT_WEIGHTS; i++) myOutput_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * dstd;
  for (int i = 0; i < NUM_CLASSES; i++) myOutput_b[i] = 0;
  Serial.println("Random starting weights set");
}

// Bytes of PSRAM that the training-only buffers need.
size_t myTrainingBytesNeeded() {
  size_t floats = (size_t)(CONV1_WEIGHTS + CONV1_FILTERS + CONV2_WEIGHTS + CONV2_FILTERS + OUTPUT_WEIGHTS + NUM_CLASSES) * 3;  // gradient + Adam m + Adam v
  floats += (size_t)FLATTENED_SIZE * 2 + POOL1_OUT_FLOATS + CONV1_OUT_FLOATS;                                                  // back-propagation buffers
  return floats * sizeof(float);
}

// Allocates the training buffers once. Returns false (and frees anything partly allocated)
// when there is not enough PSRAM. Inference is never affected.
bool myAllocateTrainingBuffers() {
  if (myTrainingBuffersReady) return true;

  size_t need = myTrainingBytesNeeded();
  if (ESP.getFreePsram() < need + 32768) {
    Serial.printf("Training needs about %u KB of PSRAM but only %u KB is free.\n",
                  (unsigned)(need / 1024), (unsigned)(ESP.getFreePsram() / 1024));
    return false;
  }

  MyBufSpec specs[] = {
    { &myConv1_w_grad,  CONV1_WEIGHTS,  false }, { &myConv1_b_grad,  CONV1_FILTERS, false },
    { &myConv2_w_grad,  CONV2_WEIGHTS,  false }, { &myConv2_b_grad,  CONV2_FILTERS, false },
    { &myOutput_w_grad, OUTPUT_WEIGHTS, false }, { &myOutput_b_grad, NUM_CLASSES,   false },
    { &myConv1_w_m, CONV1_WEIGHTS,  true }, { &myConv1_w_v, CONV1_WEIGHTS,  true },
    { &myConv1_b_m, CONV1_FILTERS,  true }, { &myConv1_b_v, CONV1_FILTERS,  true },
    { &myConv2_w_m, CONV2_WEIGHTS,  true }, { &myConv2_w_v, CONV2_WEIGHTS,  true },
    { &myConv2_b_m, CONV2_FILTERS,  true }, { &myConv2_b_v, CONV2_FILTERS,  true },
    { &myOutput_w_m, OUTPUT_WEIGHTS, true }, { &myOutput_w_v, OUTPUT_WEIGHTS, true },
    { &myOutput_b_m, NUM_CLASSES,    true }, { &myOutput_b_v, NUM_CLASSES,    true },
    { &myDense_grad, FLATTENED_SIZE,    false },
    { &myConv2_grad, FLATTENED_SIZE,    false },
    { &myPool1_grad, POOL1_OUT_FLOATS,  false },
    { &myConv1_grad, CONV1_OUT_FLOATS,  false }
  };
  const int n = sizeof(specs) / sizeof(specs[0]);

  bool ok = true;
  for (int i = 0; i < n; i++) {
    *specs[i].ptr = specs[i].zero ? (float*)ps_calloc(specs[i].count, sizeof(float))
                                  : (float*)ps_malloc(specs[i].count * sizeof(float));
    if (!*specs[i].ptr) { ok = false; break; }
  }
  if (!ok) {
    Serial.println("Training buffer allocation failed - freeing what was allocated");
    for (int i = 0; i < n; i++) {
      if (*specs[i].ptr) { free(*specs[i].ptr); *specs[i].ptr = nullptr; }
    }
    return false;
  }

  myTrainingBuffersReady = true;
  Serial.printf("Training buffers allocated. Free PSRAM: %u KB\n", (unsigned)(ESP.getFreePsram() / 1024));
  return true;
}

// Forget the optimizer memory (done at the start of every training run).
void myResetAdam() {
  memset(myConv1_w_m, 0, CONV1_WEIGHTS * sizeof(float));  memset(myConv1_w_v, 0, CONV1_WEIGHTS * sizeof(float));
  memset(myConv1_b_m, 0, CONV1_FILTERS * sizeof(float));  memset(myConv1_b_v, 0, CONV1_FILTERS * sizeof(float));
  memset(myConv2_w_m, 0, CONV2_WEIGHTS * sizeof(float));  memset(myConv2_w_v, 0, CONV2_WEIGHTS * sizeof(float));
  memset(myConv2_b_m, 0, CONV2_FILTERS * sizeof(float));  memset(myConv2_b_v, 0, CONV2_FILTERS * sizeof(float));
  memset(myOutput_w_m, 0, OUTPUT_WEIGHTS * sizeof(float)); memset(myOutput_w_v, 0, OUTPUT_WEIGHTS * sizeof(float));
  memset(myOutput_b_m, 0, NUM_CLASSES * sizeof(float));   memset(myOutput_b_v, 0, NUM_CLASSES * sizeof(float));
}


// =============================================================================
//  SAVING AND LOADING WEIGHTS
//  File order (all 32-bit floats): conv1 weights, conv1 bias, conv2 weights,
//  conv2 bias, output weights, output bias. The web page uses the same order.
// =============================================================================
static void myDumpArray(File& file, const char* name, const float* data, int size) {
  file.printf("const float %s[] = { ", name);
  for (int i = 0; i < size; i++) {
    file.print(data[i], 6);
    file.print("f");
    if (i < size - 1) file.print(", ");
    if ((i + 1) % 8 == 0) file.println();
  }
  file.println(" };");
}

// Writes /header/myWeights.h so the weights can be baked into the sketch.
void myExportHeader() {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot export header");
    return;
  }
  if (!SD.exists("/header")) SD.mkdir("/header");
  File file = SD.open("/header/myWeights.h", FILE_WRITE);
  if (!file) return;
  file.println("#ifndef MY_MODEL_H\n#define MY_MODEL_H");
  file.println("// ======================================================");
  file.println("// After copying this file to your sketch folder, make the");
  file.println("// following lines in the main sketch match the training:");
  file.println("//");
  file.printf( "//   #define INPUT_SIZE %d\n", INPUT_SIZE);
  file.printf( "//   #define CONV1_FILTERS %d\n", CONV1_FILTERS);
  file.printf( "//   #define CONV2_FILTERS %d\n", CONV2_FILTERS);
  file.printf( "//   #define NUM_CLASSES %d\n", NUM_CLASSES);
  file.print(  "//   String myClassLabels[NUM_CLASSES] = {");
  for (int i = 0; i < NUM_CLASSES; i++) {
    file.printf("\"%s\"", myClassLabels[i].c_str());
    if (i < NUM_CLASSES - 1) file.print(", ");
  }
  file.println("};");
  file.println("//");
  file.println("// Then remove the // in front of:  #define USE_BAKED_WEIGHTS");
  file.println("// ======================================================");
  myDumpArray(file, "myModel_conv1_w",  myConv1_w,  CONV1_WEIGHTS);
  myDumpArray(file, "myModel_conv1_b",  myConv1_b,  CONV1_FILTERS);
  myDumpArray(file, "myModel_conv2_w",  myConv2_w,  CONV2_WEIGHTS);
  myDumpArray(file, "myModel_conv2_b",  myConv2_b,  CONV2_FILTERS);
  myDumpArray(file, "myModel_output_w", myOutput_w, OUTPUT_WEIGHTS);
  myDumpArray(file, "myModel_output_b", myOutput_b, NUM_CLASSES);
  file.println("#endif");
  file.close();
  Serial.println("Wrote /header/myWeights.h (copy it to the sketch folder to bake the weights in)");
}

// Loads /header/myWeights.bin. Refuses a file whose size does not match this sketch's layout.
bool myLoadWeights() {
  if (!mySDavailable) {
    Serial.println("No SD card - skipping weight load");
    return false;
  }
  if (!SD.exists("/header/myWeights.bin")) {
    Serial.println("No SD weights file found");
    return false;
  }
  Serial.println("Loading weights from SD...");
  File f = SD.open("/header/myWeights.bin", FILE_READ);
  if (!f) return false;

  if ((size_t)f.size() != MY_EXPECTED_WEIGHT_BYTES) {
    Serial.printf("REFUSED myWeights.bin: file is %u bytes but this sketch needs %u bytes\n",
                  (unsigned)f.size(), (unsigned)MY_EXPECTED_WEIGHT_BYTES);
    Serial.printf("Sketch layout: INPUT_SIZE %d, CONV1_FILTERS %d, CONV2_FILTERS %d, NUM_CLASSES %d\n",
                  INPUT_SIZE, CONV1_FILTERS, CONV2_FILTERS, NUM_CLASSES);
    Serial.println("Make these four settings match the web page, reflash, or retrain.");
    f.close();
    return false;
  }

  f.read((uint8_t*)myConv1_w,  CONV1_WEIGHTS * 4);
  f.read((uint8_t*)myConv1_b,  CONV1_FILTERS * 4);
  f.read((uint8_t*)myConv2_w,  CONV2_WEIGHTS * 4);
  f.read((uint8_t*)myConv2_b,  CONV2_FILTERS * 4);
  f.read((uint8_t*)myOutput_w, OUTPUT_WEIGHTS * 4);
  f.read((uint8_t*)myOutput_b, NUM_CLASSES * 4);
  f.close();
  Serial.println("Weights loaded successfully");
  myWeightsTrained = true;
  return true;
}

void mySaveWeights() {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot save weights");
    return;
  }
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/myWeights.bin", FILE_WRITE);
  if (f) {
    f.write((uint8_t*)myConv1_w,  CONV1_WEIGHTS * 4);
    f.write((uint8_t*)myConv1_b,  CONV1_FILTERS * 4);
    f.write((uint8_t*)myConv2_w,  CONV2_WEIGHTS * 4);
    f.write((uint8_t*)myConv2_b,  CONV2_FILTERS * 4);
    f.write((uint8_t*)myOutput_w, OUTPUT_WEIGHTS * 4);
    f.write((uint8_t*)myOutput_b, NUM_CLASSES * 4);
    f.close();
    Serial.println("Weights saved to SD");
  }
  myExportHeader();
}


// =============================================================================
//  READ SETTINGS FROM /header/config.json  (written by the web page)
//  Only the "classes" list is used, and only when it has exactly NUM_CLASSES names.
//  The labels compiled into the sketch are the fallback. The layout values in the
//  file are only compared with the sketch and a WARNING is printed on a mismatch.
// =============================================================================

// Finds  "key": 123  in the text. Returns true and sets out when found.
static bool myJsonInt(const String& t, const char* key, int& out) {
  String k("\"");
  k += key;
  k += "\"";
  int p = t.indexOf(k.c_str());
  if (p < 0) return false;
  p = t.indexOf(':', p);
  if (p < 0) return false;
  p++;
  int len = (int)t.length();
  while (p < len && (t[p] == ' ' || t[p] == '\n' || t[p] == '\r' || t[p] == '\t')) p++;
  bool neg = false;
  if (p < len && t[p] == '-') { neg = true; p++; }
  if (p >= len || t[p] < '0' || t[p] > '9') return false;
  long v = 0;
  while (p < len && t[p] >= '0' && t[p] <= '9') { v = v * 10 + (t[p] - '0'); p++; }
  out = (int)(neg ? -v : v);
  return true;
}

// Reads the list of strings named key. Returns how many were found, or -1 if the key or list is missing.
static int myJsonStringArray(const String& t, const char* key, std::vector<String>& out) {
  String k("\"");
  k += key;
  k += "\"";
  int p = t.indexOf(k.c_str());
  if (p < 0) return -1;
  p = t.indexOf('[', p);
  if (p < 0) return -1;
  p++;
  int len = (int)t.length();
  out.clear();
  while (p < len) {
    char c = t[p];
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == ',') { p++; continue; }
    if (c == ']') break;
    if (c != '"') return -1;                  // not a list of strings
    p++;
    String s("");
    while (p < len && t[p] != '"') {
      if (t[p] == '\\' && p + 1 < len) p++;   // keep the escaped character
      s += t[p];
      p++;
    }
    if (p >= len) return -1;                  // string never closed
    p++;                                      // closing quote
    out.push_back(s);
  }
  return (int)out.size();
}

void myLoadConfig() {
  if (!mySDavailable) return;
  if (!SD.exists("/header/config.json")) {
    Serial.println("No /header/config.json - using the class labels compiled into the sketch");
    return;
  }
  File f = SD.open("/header/config.json", FILE_READ);
  if (!f) return;
  size_t sz = f.size();
  if (sz == 0 || sz > 4096) {
    Serial.printf("config.json ignored: unexpected size %u bytes\n", (unsigned)sz);
    f.close();
    return;
  }
  String txt;
  txt.reserve(sz + 1);
  while (f.available()) txt += (char)f.read();
  f.close();

  std::vector<String> names;
  int n = myJsonStringArray(txt, "classes", names);
  if (n < 0) {
    Serial.println("config.json has no readable \"classes\" list - keeping the compiled class labels");
  } else if (n != NUM_CLASSES) {
    Serial.printf("config.json lists %d classes but the sketch has NUM_CLASSES %d - keeping the compiled labels.\n", n, NUM_CLASSES);
    Serial.println("To add or remove classes, change NUM_CLASSES and myClassLabels[] in the sketch and reflash.");
  } else {
    Serial.println("Class labels loaded from /header/config.json:");
    for (int i = 0; i < NUM_CLASSES; i++) {
      myClassLabels[i] = names[i];
      Serial.printf("  %d: %s\n", i, myClassLabels[i].c_str());
    }
  }

  int v;
  if (myJsonInt(txt, "input_size", v) && v != INPUT_SIZE)
    Serial.printf("WARNING: config.json input_size %d but the sketch INPUT_SIZE is %d\n", v, INPUT_SIZE);
  if (myJsonInt(txt, "conv1_filters", v) && v != CONV1_FILTERS)
    Serial.printf("WARNING: config.json conv1_filters %d but the sketch CONV1_FILTERS is %d\n", v, CONV1_FILTERS);
  if (myJsonInt(txt, "conv2_filters", v) && v != CONV2_FILTERS)
    Serial.printf("WARNING: config.json conv2_filters %d but the sketch CONV2_FILTERS is %d\n", v, CONV2_FILTERS);
}


// =============================================================================
//  WEB PAGE DEBUG FRAMES
//  The page sends 'D' when it connects and every 5 s, and 'd' when it disconnects.
//  While a 'D' was seen in the last 15 s the device prints lines that start with "@F":
//    @F <kind> <n> <pred> <probs|-> <logits|-> <layout> <centre RGB|-> <heatSide> <heat base64|-> <jpeg base64>
//  kind I = inference frame (every 10th), C = image just saved, P = live preview while collecting.
//  The JPEG is exactly what the camera produced. The heatmap is the maximum over the
//  conv2 filters, scaled 0..255.
// =============================================================================

// Returns true when the character was a debug command (so the caller ignores it).
bool myHandleDebugChar(char c) {
  if (c == 'D') {
    if (!myDebugStream) Serial.println("Debug frames ON");
    myDebugStream = true;
    myDebugLastSeen = millis();
    return true;
  }
  if (c == 'd') {
    if (myDebugStream) Serial.println("Debug frames OFF");
    myDebugStream = false;
    return true;
  }
  return false;
}

// Prints bytes as base64 in chunks of 384 bytes (a multiple of 3, so the pieces join cleanly).
static void myPrintB64(const uint8_t* d, size_t n) {
  unsigned char out[520];
  while (n > 0) {
    size_t take = n > 384 ? 384 : n;
    size_t ol = 0;
    mbedtls_base64_encode(out, sizeof(out), &ol, d, take);
    Serial.write(out, ol);
    d += take;
    n -= take;
  }
}

void myDebugSendFrame(char kind, int n, camera_fb_t* fb, int pred, const float* logits) {
  if (!myDebugStream || !fb || !Serial) return;
  if (millis() - myDebugLastSeen > 15000) { myDebugStream = false; return; }   // the page stopped sending
  const bool inf = (kind == 'I' && logits != nullptr);
  Serial.printf("@F %c %d %d ", kind, n, pred);
  if (inf) {
    for (int i = 0; i < NUM_CLASSES; i++) { if (i) Serial.print(','); Serial.print(myDense_output[i], 4); }
    Serial.print(' ');
    for (int i = 0; i < NUM_CLASSES; i++) { if (i) Serial.print(','); Serial.print(logits[i], 4); }
  } else {
    Serial.print("- -");
  }
  Serial.printf(" %dx%dx%d ", INPUT_SIZE, CONV1_FILTERS, CONV2_FILTERS);
  if (inf) {
    int c = ((INPUT_SIZE / 2) * INPUT_SIZE + INPUT_SIZE / 2) * 3;    // centre pixel of the model input
    Serial.print(myInputBuffer[c], 4); Serial.print(',');
    Serial.print(myInputBuffer[c + 1], 4); Serial.print(',');
    Serial.print(myInputBuffer[c + 2], 4);
  } else {
    Serial.print('-');
  }
  Serial.print(' ');
  if (inf) {
    const int hn = CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE;
    static uint8_t heat[CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE];
    float lo = 1e30f, hi = -1e30f;
    for (int i = 0; i < hn; i++) {
      float m = myConv2_output[i];
      for (int f = 1; f < CONV2_FILTERS; f++) { float v = myConv2_output[f * hn + i]; if (v > m) m = v; }
      if (m < lo) lo = m;
      if (m > hi) hi = m;
    }
    float span = hi - lo;
    if (span < 1e-9f) span = 1.0f;
    for (int i = 0; i < hn; i++) {
      float m = myConv2_output[i];
      for (int f = 1; f < CONV2_FILTERS; f++) { float v = myConv2_output[f * hn + i]; if (v > m) m = v; }
      heat[i] = (uint8_t)(255.0f * (m - lo) / span + 0.5f);
    }
    Serial.printf("%d ", CONV2_OUTPUT_SIZE);
    myPrintB64(heat, hn);
  } else {
    Serial.print("0 -");
  }
  Serial.print(' ');
  myPrintB64(fb->buf, fb->len);
  Serial.println();
}


// =============================================================================
//  IMAGES
// =============================================================================

// Shrinks the 240x240 RGB picture in rgb240 to INPUT_SIZE x INPUT_SIZE floats (0..1).
// Training and inference both use this one function, so they see identical input.
void myResizeToInput(const uint8_t* rgb240, float* dst) {
  for (int y = 0; y < INPUT_SIZE; y++) {
    int rowOffset = myResizeLut[y] * 240;
    for (int x = 0; x < INPUT_SIZE; x++) {
      int srcIdx = (rowOffset + myResizeLut[x]) * 3;
      int dstIdx = (y * INPUT_SIZE + x) * 3;
      dst[dstIdx]     = rgb240[srcIdx]     * 0.003921569f;    // divide by 255
      dst[dstIdx + 1] = rgb240[srcIdx + 1] * 0.003921569f;
      dst[dstIdx + 2] = rgb240[srcIdx + 2] * 0.003921569f;
    }
  }
}

// Reads one JPEG from the SD card and fills buf with the model input.
bool myLoadImageFromFile(const char* path, float* buf) {
  File f = SD.open(path);
  if (!f) return false;

  size_t sz = f.size();
  uint8_t* jpg = (uint8_t*)ps_malloc(sz);
  if (!jpg) { f.close(); return false; }
  f.read(jpg, sz);
  f.close();

  if (!myRgbBuffer) { free(jpg); return false; }
  bool ok = fmt2rgb888(jpg, sz, PIXFORMAT_JPEG, myRgbBuffer);
  free(jpg);
  if (!ok) return false;

  myResizeToInput(myRgbBuffer, buf);
  return true;
}

// A file name inside folder that does not exist yet (so a new photo never overwrites an old one).
String myUniqueImagePath(const String& folder) {
  unsigned long n = millis();
  String p = folder + "/img_" + String(n) + ".jpg";
  while (SD.exists(p)) {
    n++;
    p = folder + "/img_" + String(n) + ".jpg";
  }
  return p;
}


// =============================================================================
//  MENU ITEM 1..NUM_CLASSES : COLLECT IMAGES
//  Shows a live preview on the OLED and saves 240x240 JPEGs into /images/<class>/.
//  A tap (or 't') takes a picture. 3+ taps (or 'l') returns to the menu.
// =============================================================================
void myActionCollect(int classIdx) {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot collect images");
    myShowMessage("No SD card");
    delay(2000);
    myResetMenuState();
    return;
  }
  if (!myCameraOk) {
    Serial.println("Camera not working - cannot collect images");
    myShowMessage("Camera error");
    delay(2000);
    myResetMenuState();
    return;
  }

  Serial.printf("\n>>> Collection mode: %s\n", myClassLabels[classIdx].c_str());
  Serial.println("  TAP (1-2 taps) = capture image      LONG PRESS (3+ taps) = exit to menu");
  Serial.println("  Serial: 't' = capture, 'l' = exit");

  myResetTouchState();

  String folder = "/images/" + myClassLabels[classIdx];
  if (!SD.exists("/images")) SD.mkdir("/images");
  if (!SD.exists(folder)) SD.mkdir(folder);

  // Count the images already in this class folder.
  int imageCount = 0;
  File root = SD.open(folder);
  if (root) {
    while (File file = root.openNextFile()) {
      String fn = String(file.name());
      if (!file.isDirectory() && (fn.endsWith(".jpg") || fn.endsWith(".JPG"))) imageCount++;
      file.close();
    }
    root.close();
  }
  Serial.printf("This class already has %d images\n", imageCount);

  unsigned long lastCameraDrain = 0;    // last time the camera buffer was serviced
  unsigned long lastOLED = 0;           // last time the OLED preview was refreshed
  unsigned long lastDebugPreview = 0;   // last time a preview went to the web page
  bool oledNeedsUpdate = false;
  bool shouldCapture = false;

  while (true) {
    unsigned long now = millis();

    // Every 50 ms take a frame out of the camera so its buffer never overflows.
    if (now - lastCameraDrain > 50) {
      lastCameraDrain = now;

      if (!shouldCapture) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) {
          if (myDebugStream && now - lastDebugPreview > 1000) {
            lastDebugPreview = now;
            myDebugSendFrame('P', imageCount, fb, -1, nullptr);
          }
          // Decoding to RGB is slow, so only do it when the OLED is due for a refresh.
          if (now - lastOLED > 250 && myRgbBuffer) {
            if (fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) {
              oledNeedsUpdate = true;
              lastOLED = now;
            }
          }
          esp_camera_fb_return(fb);
        }
      }
    }

    if (oledNeedsUpdate) {
      oledNeedsUpdate = false;
      myRenderRgbToOLED(-1);            // -1 = LIVE badge
    }

    // Serial keys.
    if (Serial.available()) {
      char c = Serial.read();
      if (myHandleDebugChar(c)) {
        // handled: the web page is asking for debug frames
      } else if (c == 'l' || c == 'L') {
        myResetMenuState();
        return;
      } else if (c == 't' || c == 'T') {
        shouldCapture = true;
      }
    }

    // Touch pad.
    int touchAction = myCheckTouchInput();
    if (touchAction == 2) {
      Serial.println("Exiting collection mode");
      myResetMenuState();
      return;
    } else if (touchAction == 1) {
      shouldCapture = true;
    }

    // Save one picture.
    if (shouldCapture) {
      shouldCapture = false;
      camera_fb_t* fb = esp_camera_fb_get();
      if (fb) {
        String fileName = myUniqueImagePath(folder);
        File file = SD.open(fileName, FILE_WRITE);
        if (file) {
          file.write(fb->buf, fb->len);
          file.close();
          imageCount++;
          Serial.printf("Saved: %s (Total: %d)\n", fileName.c_str(), imageCount);
          myDisplayImageOnOLED(fb, imageCount);
          myDebugSendFrame('C', imageCount, fb, -1, nullptr);
          delay(300);                   // let the snapshot show before the live view returns
          lastOLED = millis();
        }
        esp_camera_fb_return(fb);
      }
    }

    delay(5);
  }
}


// =============================================================================
//  THE CNN
//    input 3 x INPUT_SIZE x INPUT_SIZE (RGB, 0..1)
//    -> conv 3x3 (CONV1_FILTERS) -> leaky ReLU -> 2x2 max pool
//    -> conv 3x3 (CONV2_FILTERS) -> leaky ReLU
//    -> flatten -> dense (NUM_CLASSES) -> softmax
//  Data layout is "channel first" per filter: [filter][row][column].
// =============================================================================

// Runs the network on input. Probabilities end up in myDense_output, raw scores in logits.
void myForwardPass(float* input, float* logits) {
  // Conv1: INPUT_SIZE x INPUT_SIZE x 3 -> CONV1_OUTPUT_SIZE^2 x CONV1_FILTERS
  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ob = f * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE;
    for (int y = 0; y < CONV1_OUTPUT_SIZE; y++) {
      for (int x = 0; x < CONV1_OUTPUT_SIZE; x++) {
        float sum = 0;
        for (int ky = 0; ky < 3; ky++) {
          for (int kx = 0; kx < 3; kx++) {
            int inPos = ((y + ky) * INPUT_SIZE + (x + kx)) * 3;
            int wPos = f * CONV1_IN_STRIDE + ky * 9 + kx * 3;
            sum += input[inPos]     * myConv1_w[wPos] +
                   input[inPos + 1] * myConv1_w[wPos + 1] +
                   input[inPos + 2] * myConv1_w[wPos + 2];
          }
        }
        myConv1_output[ob + y * CONV1_OUTPUT_SIZE + x] = leaky_relu(clip_value(sum + myConv1_b[f]));
      }
    }
  }

  // Pool1: keep the largest of each 2x2 block
  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ib = f * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE, ob = f * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE;
    for (int y = 0; y < POOL1_OUTPUT_SIZE; y++) {
      for (int x = 0; x < POOL1_OUTPUT_SIZE; x++) {
        int iy = y * 2, ix = x * 2;
        float maxVal = myConv1_output[ib + iy * CONV1_OUTPUT_SIZE + ix];
        maxVal = max(maxVal, myConv1_output[ib + iy * CONV1_OUTPUT_SIZE + ix + 1]);
        maxVal = max(maxVal, myConv1_output[ib + (iy + 1) * CONV1_OUTPUT_SIZE + ix]);
        maxVal = max(maxVal, myConv1_output[ib + (iy + 1) * CONV1_OUTPUT_SIZE + ix + 1]);
        myPool1_output[ob + y * POOL1_OUTPUT_SIZE + x] = maxVal;
      }
    }
  }

  // Conv2: POOL1_OUTPUT_SIZE^2 x CONV1_FILTERS -> CONV2_OUTPUT_SIZE^2 x CONV2_FILTERS
  for (int f = 0; f < CONV2_FILTERS; f++) {
    int ob = f * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE;
    for (int y = 0; y < CONV2_OUTPUT_SIZE; y++) {
      for (int x = 0; x < CONV2_OUTPUT_SIZE; x++) {
        float sum = 0;
        for (int c = 0; c < CONV1_FILTERS; c++) {
          int ib = c * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE;
          for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
              sum += myPool1_output[ib + (y + ky) * POOL1_OUTPUT_SIZE + (x + kx)] *
                     myConv2_w[f * CONV2_IN_STRIDE + c * 9 + ky * 3 + kx];
            }
          }
        }
        myConv2_output[ob + y * CONV2_OUTPUT_SIZE + x] = leaky_relu(clip_value(sum + myConv2_b[f]));
      }
    }
  }

  // Dense layer (Kahan summation keeps the long sum accurate)
  for (int c = 0; c < NUM_CLASSES; c++) {
    double sum = 0, comp = 0;
    for (int i = 0; i < FLATTENED_SIZE; i++) {
      double term = myConv2_output[i] * myOutput_w[c * FLATTENED_SIZE + i];
      double y = term - comp;
      double t = sum + y;
      comp = (t - sum) - y;
      sum = t;
    }
    myDense_output[c] = clip_value((float)sum + myOutput_b[c], -50, 50);
  }

  // Softmax: turn the scores into probabilities that add up to 1
  float mx = myDense_output[0];
  for (int i = 1; i < NUM_CLASSES; i++) mx = max(mx, myDense_output[i]);
  float expSum = 0;
  for (int i = 0; i < NUM_CLASSES; i++) expSum += exp(myDense_output[i] - mx);
  for (int i = 0; i < NUM_CLASSES; i++) {
    logits[i] = myDense_output[i];
    myDense_output[i] = exp(myDense_output[i] - mx) / expSum;
  }
}

// ---- Backward pass: how much should each weight change? -----------------------
// Weight gradients are ADDED (+=) over the images of a batch. They are cleared once
// per batch in myActionTrain(). The "propagation" buffers are cleared for every image.

void myBackwardDense(int label) {
  memset(myDense_grad, 0, FLATTENED_SIZE * sizeof(float));
  for (int c = 0; c < NUM_CLASSES; c++) {
    float error = myDense_output[c] - (c == label ? 1.0f : 0.0f);
    for (int i = 0; i < FLATTENED_SIZE; i++) {
      myOutput_w_grad[c * FLATTENED_SIZE + i] += error * myConv2_output[i];
      myDense_grad[i] += error * myOutput_w[c * FLATTENED_SIZE + i];
    }
    myOutput_b_grad[c] += error;
  }
}

void myBackwardConv2() {
  for (int i = 0; i < FLATTENED_SIZE; i++) {
    myConv2_grad[i] = myDense_grad[i] * leaky_relu_deriv(myConv2_output[i]);
  }
  memset(myPool1_grad, 0, POOL1_OUT_FLOATS * sizeof(float));

  for (int f = 0; f < CONV2_FILTERS; f++) {
    int ob = f * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE;
    for (int y = 0; y < CONV2_OUTPUT_SIZE; y++) {
      for (int x = 0; x < CONV2_OUTPUT_SIZE; x++) {
        float grad = myConv2_grad[ob + y * CONV2_OUTPUT_SIZE + x];
        myConv2_b_grad[f] += grad;
        for (int c = 0; c < CONV1_FILTERS; c++) {
          int ib = c * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE;
          for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
              int pi = ib + (y + ky) * POOL1_OUTPUT_SIZE + (x + kx);
              int wi = f * CONV2_IN_STRIDE + c * 9 + ky * 3 + kx;
              myConv2_w_grad[wi] += grad * myPool1_output[pi];
              myPool1_grad[pi] += grad * myConv2_w[wi];
            }
          }
        }
      }
    }
  }
}

// The gradient only flows back to the pixel that "won" each 2x2 max pool.
void myBackwardPool1() {
  memset(myConv1_grad, 0, CONV1_OUT_FLOATS * sizeof(float));
  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ib = f * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE, ob = f * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE;
    for (int y = 0; y < POOL1_OUTPUT_SIZE; y++) {
      for (int x = 0; x < POOL1_OUTPUT_SIZE; x++) {
        int iy = y * 2, ix = x * 2;
        float poolVal = myPool1_output[ob + y * POOL1_OUTPUT_SIZE + x];
        float grad = myPool1_grad[ob + y * POOL1_OUTPUT_SIZE + x];
        if (myConv1_output[ib + iy * CONV1_OUTPUT_SIZE + ix] == poolVal)             myConv1_grad[ib + iy * CONV1_OUTPUT_SIZE + ix] += grad;
        if (myConv1_output[ib + iy * CONV1_OUTPUT_SIZE + ix + 1] == poolVal)         myConv1_grad[ib + iy * CONV1_OUTPUT_SIZE + ix + 1] += grad;
        if (myConv1_output[ib + (iy + 1) * CONV1_OUTPUT_SIZE + ix] == poolVal)       myConv1_grad[ib + (iy + 1) * CONV1_OUTPUT_SIZE + ix] += grad;
        if (myConv1_output[ib + (iy + 1) * CONV1_OUTPUT_SIZE + ix + 1] == poolVal)   myConv1_grad[ib + (iy + 1) * CONV1_OUTPUT_SIZE + ix + 1] += grad;
      }
    }
  }
}

void myBackwardConv1() {
  for (int i = 0; i < CONV1_OUT_FLOATS; i++) {
    myConv1_grad[i] *= leaky_relu_deriv(myConv1_output[i]);
  }

  for (int f = 0; f < CONV1_FILTERS; f++) {
    int ob = f * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE;
    for (int y = 0; y < CONV1_OUTPUT_SIZE; y++) {
      for (int x = 0; x < CONV1_OUTPUT_SIZE; x++) {
        float grad = myConv1_grad[ob + y * CONV1_OUTPUT_SIZE + x];
        myConv1_b_grad[f] += grad;
        for (int ky = 0; ky < 3; ky++) {
          for (int kx = 0; kx < 3; kx++) {
            int inPos = ((y + ky) * INPUT_SIZE + (x + kx)) * 3;
            int wPos = f * CONV1_IN_STRIDE + ky * 9 + kx * 3;
            myConv1_w_grad[wPos]     += grad * myInputBuffer[inPos];
            myConv1_w_grad[wPos + 1] += grad * myInputBuffer[inPos + 1];
            myConv1_w_grad[wPos + 2] += grad * myInputBuffer[inPos + 2];
          }
        }
      }
    }
  }
}

// ---- Optimizer: Adam ----------------------------------------------------------
// Moves each weight against its gradient, with a smoothed step size per weight.
void myAdamUpdate(float* w, float* g, float* m, float* v, int size, int step) {
  float b1 = 0.9f, b2 = 0.999f, eps = 1e-6f;     // eps 1e-6 (not 1e-8) avoids NaN in 32-bit floats
  float lr_t = LEARNING_RATE * sqrt(1 - pow(b2, step)) / (1 - pow(b1, step));
  for (int i = 0; i < size; i++) {
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    w[i] -= lr_t * m[i] / (sqrt(v[i]) + eps);
    w[i] = clip_value(w[i], -10, 10);
  }
}

void myUpdateWeights(int step) {
  myAdamUpdate(myConv1_w,  myConv1_w_grad,  myConv1_w_m,  myConv1_w_v,  CONV1_WEIGHTS,  step);
  myAdamUpdate(myConv1_b,  myConv1_b_grad,  myConv1_b_m,  myConv1_b_v,  CONV1_FILTERS,  step);
  myAdamUpdate(myConv2_w,  myConv2_w_grad,  myConv2_w_m,  myConv2_w_v,  CONV2_WEIGHTS,  step);
  myAdamUpdate(myConv2_b,  myConv2_b_grad,  myConv2_b_m,  myConv2_b_v,  CONV2_FILTERS,  step);
  myAdamUpdate(myOutput_w, myOutput_w_grad, myOutput_w_m, myOutput_w_v, OUTPUT_WEIGHTS, step);
  myAdamUpdate(myOutput_b, myOutput_b_grad, myOutput_b_m, myOutput_b_v, NUM_CLASSES,    step);
}


// =============================================================================
//  MENU ITEM NUM_CLASSES+1 : TRAIN ON THE DEVICE
//  Trains on the images in /images/<class>/, saves myWeights.bin and myWeights.h.
//  If the model is too big for the device's memory it says so and returns to the
//  menu; Infer still works with weights trained on the web page.
// =============================================================================

// True when the user asked to stop (serial 'x' or 'l', or 3+ taps on the touch pad).
bool myTrainingExitRequested() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'x' || c == 'X' || c == 'l' || c == 'L') {
      Serial.println("Stopping training...");
      return true;
    }
  }
  myCheckTouchBackground();
  if (myPeekTouchAction() == 2) {
    myCheckTouchInput();                // consume the long press
    Serial.println("Long press - stopping training");
    return true;
  }
  return false;
}

void myActionTrain() {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot train");
    myShowMessage("No SD card");
    delay(2000);
    myResetMenuState();
    return;
  }

  if (!myAllocateTrainingBuffers()) {
    Serial.println("Not enough memory to train this model size on the device.");
    Serial.println("Train on the web page instead (or lower INPUT_SIZE / filters). Infer still works.");
    myShowMessage("Not enough", "memory to train", "Infer still OK");
    delay(3000);
    myResetMenuState();
    return;
  }

  Serial.println("\n>>> Training mode");
  Serial.println("  During training: 3+ taps (or 'l') = save and exit");
  Serial.println("  When finished:   TAP (or 't') = train again, 3+ taps (or 'l') = exit");

  myResetTouchState();
  myShowMessage("TRAINING MODE", "Loading...");

  if (myLoadWeights()) {
    Serial.println("Continuing from saved weights");
  } else {
    Serial.println("Starting fresh training");
  }

  while (true) {
    // ---- Build the list of training images ----
    myTrainingData.clear();
    for (int i = 0; i < NUM_CLASSES; i++) {
      File root = SD.open("/images/" + myClassLabels[i]);
      if (root) {
        while (File file = root.openNextFile()) {
          if (!file.isDirectory()) {
            String fn = String(file.name());
            if (fn.endsWith(".jpg") || fn.endsWith(".JPG")) {
              myTrainingData.push_back({file.path(), i});
            }
          }
          file.close();
        }
        root.close();
      }
    }

    if (myTrainingData.empty()) {
      Serial.println("No images found in /images/<class>/");
      myShowMessage("No Images!");
      delay(2000);
      myResetMenuState();
      return;
    }

    // Sort by path so the split is always the same, then hold out the last
    // VALIDATION_IMAGES images of each class to measure accuracy on unseen pictures.
    std::sort(myTrainingData.begin(), myTrainingData.end(),
              [](const TrainingItem& a, const TrainingItem& b) { return a.path < b.path; });

    std::vector<TrainingItem> myValidationData;
    if (VALIDATION_IMAGES > 0) {
      int counts[NUM_CLASSES] = {};
      for (auto& item : myTrainingData) counts[item.label]++;
      int skip[NUM_CLASSES];
      for (int c = 0; c < NUM_CLASSES; c++) skip[c] = min(VALIDATION_IMAGES, counts[c]);

      std::vector<TrainingItem> trainOnly;
      int seen[NUM_CLASSES] = {};
      for (int i = (int)myTrainingData.size() - 1; i >= 0; i--) {
        int c = myTrainingData[i].label;
        if (seen[c] < skip[c]) {
          myValidationData.push_back(myTrainingData[i]);
          seen[c]++;
        } else {
          trainOnly.push_back(myTrainingData[i]);
        }
      }
      myTrainingData = trainOnly;
      Serial.printf("Validation: %d images  Train: %d images\n",
                    (int)myValidationData.size(), (int)myTrainingData.size());
    }

    int total = myTrainingData.size();
    int batchesPerEpoch = (total + BATCH_SIZE - 1) / BATCH_SIZE;
    int totalBatches = TARGET_EPOCHS * batchesPerEpoch;

    Serial.printf("Training: %d images, %d batches\n", total, totalBatches);

    myResetAdam();                       // the optimizer's step counter restarts, so its memory does too

    std::vector<int> indices;
    for (int i = 0; i < total; i++) indices.push_back(i);

    float runningLoss = 0;
    int lossCount = 0;

    // ---- Training loop: one batch at a time ----
    for (int batch = 0; batch < totalBatches; batch++) {
      if (myTrainingExitRequested()) {
        mySaveWeights();
        myWeightsTrained = true;
        myResetMenuState();
        return;
      }

      // Shuffle the image order at the start of each epoch.
      if (batch % batchesPerEpoch == 0) {
        int epoch = batch / batchesPerEpoch + 1;
        Serial.printf("\n--- Epoch %d/%d ---\n", epoch, TARGET_EPOCHS);
        for (int i = total - 1; i > 0; i--) {
          int j = random(i + 1);
          int tmp = indices[i];
          indices[i] = indices[j];
          indices[j] = tmp;
        }
      }

      int batchStart = (batch % batchesPerEpoch) * BATCH_SIZE;
      int batchEnd = min(batchStart + BATCH_SIZE, total);

      float batchLoss = 0;
      int correctCount = 0;

      // Clear the weight gradients once per batch; every image in the batch adds to them.
      memset(myConv1_w_grad,  0, CONV1_WEIGHTS  * sizeof(float));
      memset(myConv1_b_grad,  0, CONV1_FILTERS  * sizeof(float));
      memset(myConv2_w_grad,  0, CONV2_WEIGHTS  * sizeof(float));
      memset(myConv2_b_grad,  0, CONV2_FILTERS  * sizeof(float));
      memset(myOutput_w_grad, 0, OUTPUT_WEIGHTS * sizeof(float));
      memset(myOutput_b_grad, 0, NUM_CLASSES    * sizeof(float));

      for (int i = batchStart; i < batchEnd; i++) {
        TrainingItem& img = myTrainingData[indices[i]];

        if (!myLoadImageFromFile(img.path.c_str(), myInputBuffer)) continue;

        float logits[NUM_CLASSES];
        myForwardPass(myInputBuffer, logits);

        float loss = -log(max(myDense_output[img.label], 1e-7f));
        batchLoss += loss;

        int pred = 0;
        for (int j = 1; j < NUM_CLASSES; j++) if (myDense_output[j] > myDense_output[pred]) pred = j;
        if (pred == img.label) correctCount++;

        myBackwardDense(img.label);
        myBackwardConv2();
        myBackwardPool1();
        myBackwardConv1();

        // Check for a stop request every few images so the exit feels instant.
        if (i % 3 == 0 && myTrainingExitRequested()) {
          mySaveWeights();
          myWeightsTrained = true;
          myResetMenuState();
          return;
        }
      }

      myUpdateWeights(batch + 1);

      float avgLoss = batchLoss / (batchEnd - batchStart);
      float batchAcc = (float)correctCount / (batchEnd - batchStart);
      runningLoss += avgLoss;
      lossCount++;

      if ((batch + 1) % 5 == 0) {
        float displayLoss = runningLoss / lossCount;
        u8g2.firstPage();
        do {
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.setCursor(0, 12); u8g2.print("Training...");
          u8g2.setCursor(0, 24);
          u8g2.print("B:"); u8g2.print(batch + 1);
          u8g2.print("/"); u8g2.print(totalBatches);
          u8g2.setCursor(0, 36);
          u8g2.print("L:"); u8g2.print(displayLoss, 3);
          u8g2.print(" A:"); u8g2.print((int)(batchAcc * 100)); u8g2.print("%");
        } while (u8g2.nextPage());
        runningLoss = 0;
        lossCount = 0;
      }

      if ((batch + 1) % 10 == 0) {
        Serial.printf("Batch %d/%d - Loss: %.4f - Acc: %.1f%%\n",
                      batch + 1, totalBatches, avgLoss, batchAcc * 100);
      }
    }

    Serial.println("\n--- Training Complete ---");

    // ---- Accuracy on the held-out validation images ----
    if (!myValidationData.empty()) {
      int valCorrect = 0;
      int valCount = 0;
      for (auto& vitem : myValidationData) {
        if (!myLoadImageFromFile(vitem.path.c_str(), myInputBuffer)) continue;
        float logits[NUM_CLASSES];
        myForwardPass(myInputBuffer, logits);
        int pred = 0;
        for (int j = 1; j < NUM_CLASSES; j++)
          if (myDense_output[j] > myDense_output[pred]) pred = j;
        if (pred == vitem.label) valCorrect++;
        valCount++;
      }
      if (valCount > 0) {
        Serial.printf("Validation Accuracy: %.1f%%  (%d/%d correct)\n",
                      100.0f * valCorrect / valCount, valCorrect, valCount);
      }
    }

    mySaveWeights();
    myWeightsTrained = true;

    myShowMessage("DONE!", "Tap: again", "3+ taps: exit");
    myResetTouchState();

    Serial.println("Waiting for input...");
    Serial.println("  Serial: t = train again   l = exit");
    Serial.println("  Touch:  1-2 taps = train again   3+ taps = exit");
    while (true) {
      if (Serial.available()) {
        char c = Serial.read();
        if (c == 'x' || c == 'X' || c == 'l' || c == 'L') {
          myResetMenuState();
          return;
        } else if (c == 't' || c == 'T') {
          break;
        }
      }
      int touchAction = myCheckTouchInput();
      if (touchAction == 2) {
        myResetMenuState();
        return;
      } else if (touchAction == 1) {
        Serial.println("Starting new training cycle");
        break;
      }
      delay(10);
    }
  }
}


// =============================================================================
//  MENU ITEM NUM_CLASSES+2 : INFER (LIVE CLASSIFICATION)
//  Every camera frame is shrunk to INPUT_SIZE and run through the network.
//  Serial prints the class probabilities each frame; the OLED shows the picture
//  and the winning class every 10th frame. Exit with 't' or 'l', or touch the pad.
// =============================================================================
void myActionInfer() {
  if (!myWeightsTrained) {
    Serial.println("ERROR: no usable weights in memory.");
    Serial.println("Train first, or put a matching myWeights.bin on the SD card and reboot.");
    Serial.println("If the boot messages said REFUSED myWeights.bin, make INPUT_SIZE, CONV1_FILTERS,");
    Serial.println("CONV2_FILTERS and NUM_CLASSES match the web page and reflash.");
    myShowMessage("No weights!", "Train first or", "check boot log");
    delay(3000);
    myResetMenuState();
    return;
  }
  if (!myCameraOk) {
    Serial.println("Camera not working - cannot infer");
    myShowMessage("Camera error");
    delay(2000);
    myResetMenuState();
    return;
  }

  Serial.println("\n>>> Inference mode");
  Serial.println("  Exit: serial 't' or 'l', or touch the pad");

  myResetTouchState();

  int frameIndex = 0;                         // counts frames 0..9 for the OLED / debug update
  int pred = 0;
  unsigned long myInferCount = 0;             // frame number sent to the web page
  unsigned long inferStart = millis();

  while (true) {
    unsigned long frameStart = millis();

    if (Serial.available()) {
      char c = Serial.read();
      if (myHandleDebugChar(c)) {
        // handled: the web page is asking for debug frames
      } else if (c == 't' || c == 'T' || c == 'l' || c == 'L') {
        myResetMenuState();
        return;
      }
    }

    // A touch exits. Ignored for the first 1.5 s so the touch that opened this mode cannot close it.
    if (millis() - inferStart > 1500 && myReadTouch() > myThresholdPress) {
      Serial.println("Touch detected - exiting inference");
      delay(200);
      myResetMenuState();
      return;
    }

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera frame failed - retrying");
      delay(10);
      continue;
    }

    if (fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, myRgbBuffer)) {
      myResizeToInput(myRgbBuffer, myInputBuffer);

      float myLogits[NUM_CLASSES];
      myForwardPass(myInputBuffer, myLogits);
      myInferCount++;

      pred = 0;
      for (int i = 1; i < NUM_CLASSES; i++) {
        if (myDense_output[i] > myDense_output[pred]) pred = i;
      }

      // Every 10th frame: update the OLED and the web page (myRgbBuffer is still valid here).
      if (frameIndex == 9) {
        myDebugSendFrame('I', (int)myInferCount, fb, pred, myLogits);
        int oW = u8g2.getDisplayWidth();
        int oH = u8g2.getDisplayHeight();
        int scX = 240 / oW;
        int scY = 240 / oH;
        u8g2.firstPage();
        do {
          for (int ox = 0; ox < oW; ox++) {
            for (int oy = 0; oy < oH; oy++) {
              int pi = ((oy * scY) * 240 + (ox * scX)) * 3;
              uint8_t bright = (myRgbBuffer[pi] + myRgbBuffer[pi + 1] + myRgbBuffer[pi + 2]) / 3;
              if (bright > 100) u8g2.drawPixel(ox, oy);
            }
          }
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.setColorIndex(0);
          u8g2.drawBox(0, oH - 9, oW, 9);
          u8g2.setColorIndex(1);
          char buf[20];
          snprintf(buf, sizeof(buf), "%s %d%%", myClassLabels[pred].c_str(), (int)(myDense_output[pred] * 100));
          u8g2.drawStr(1, oH - 1, buf);
        } while (u8g2.nextPage());
      }
    }

    esp_camera_fb_return(fb);

    unsigned long frameMs = millis() - frameStart;
    Serial.printf("Frame %d: %lu ms (%.1f FPS) Current Pred: %s (%.1f%%) | All:",
                  frameIndex + 1, frameMs, 1000.0 / max(frameMs, 1UL),
                  myClassLabels[pred].c_str(), myDense_output[pred] * 100);
    for (int i = 0; i < NUM_CLASSES; i++) Serial.printf(" %.0f%%", myDense_output[i] * 100);
    Serial.println();

    frameIndex++;
    if (frameIndex >= 10) frameIndex = 0;
  }
}


// =============================================================================
//  MENU
//  Items 1..NUM_CLASSES collect images, then Train, then Infer.
//  Called from loop().
// =============================================================================

// Text for menu item 1..myTotalItems.
String myMenuLabel(int item) {
  if (item <= NUM_CLASSES) return myClassLabels[item - 1];
  if (item == NUM_CLASSES + 1) return "Train";
  return "Infer";
}

void myResetMenuState() {
  myIsSelected = false;
  myResetTouchState();
  myLastActivityTime = millis();
  myDrawMenu();
}

void myDrawMenu() {
  // ---- Serial menu ----
  Serial.println("\n=== MENU ===");
  for (int i = 1; i <= myTotalItems; i++) {
    Serial.print(i == myMenuIndex ? " > " : "   ");
    Serial.printf("%d. %s\n", i, myMenuLabel(i).c_str());
  }
  Serial.println("Keys: t = next (tap)   l = select (long press)   or press the item's number");

  // ---- OLED menu: three rows, scrolled so the chosen item is always visible ----
  int myStartItem = myMenuIndex - 1;
  if (myStartItem > myTotalItems - 2) myStartItem = myTotalItems - 2;
  if (myStartItem < 1) myStartItem = 1;

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_5x7_tf);
    u8g2.drawStr(0, 7, "TAP:next 3x:OK");
    u8g2.setFont(u8g2_font_6x10_tf);
    for (int i = 0; i < 3; i++) {
      int cur = myStartItem + i;
      if (cur > myTotalItems) break;
      String line = (cur == myMenuIndex ? "> " : "  ") + myMenuLabel(cur);
      u8g2.drawStr(0, 18 + i * 9, line.c_str());
    }
  } while (u8g2.nextPage());
}

void myExecuteMenuItem(int idx) {
  if (idx <= NUM_CLASSES)        myActionCollect(idx - 1);
  else if (idx == NUM_CLASSES + 1) myActionTrain();
  else                           myActionInfer();
}

void myHandleMenuNavigation() {
  unsigned long myCurrentMillis = millis();

  // ---- Serial keys ----
  if (!myIsSelected && Serial.available()) {
    char c = Serial.read();

    if (myHandleDebugChar(c)) {
      // handled: the web page is asking for debug frames
    }
    else if (c >= '1' && c <= '9') {            // jump straight to an item
      int newIndex = c - '0';
      if (newIndex <= myTotalItems) {
        myMenuIndex = newIndex;
        myIsSelected = true;
        myLastActivityTime = myCurrentMillis;
        myExecuteMenuItem(myMenuIndex);
      }
    }
    else if (c == 't' || c == 'T') {            // next item
      if (myCurrentMillis - myLastTapTime > myTapCooldown) {
        myMenuIndex++;
        if (myMenuIndex > myTotalItems) myMenuIndex = 1;
        myDrawMenu();
        myLastTapTime = myCurrentMillis;
        myLastActivityTime = myCurrentMillis;
      }
    }
    else if (c == 'l' || c == 'L') {            // select
      myIsSelected = true;
      myLastActivityTime = myCurrentMillis;
      myExecuteMenuItem(myMenuIndex);
    }
  }

  // ---- Touch pad ----
  if (!myIsSelected) {
    int touchAction = myCheckTouchInput();

    if (touchAction == 1) {                     // tap: next item
      if (myCurrentMillis - myLastTapTime > myTapCooldown) {
        myMenuIndex++;
        if (myMenuIndex > myTotalItems) myMenuIndex = 1;
        myDrawMenu();
        myLastTapTime = myCurrentMillis;
        myLastActivityTime = myCurrentMillis;
      }
    }
    else if (touchAction == 2) {                // 3+ taps: select
      myIsSelected = true;
      myLastActivityTime = myCurrentMillis;
      myExecuteMenuItem(myMenuIndex);
    }
  }
}
