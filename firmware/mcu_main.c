/* Cortex-M4 size build: one classification of a window held in RAM.
 * Built only to measure flash and RAM (arm-none-eabi-size, -fstack-usage);
 * it is not run on a board. On a real device the window would be filled by
 * the ADC DMA and normalised before the call. */
#include "wdcnn.h"

float window[WD_INPUT];
volatile int result;

int main(void)
{
    float logits[WD_CLASSES];
    for (;;) {
        result = wdcnn_classify(window, logits);
    }
}
