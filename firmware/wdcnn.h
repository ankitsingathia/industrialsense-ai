#ifndef WDCNN_H
#define WDCNN_H

#define WD_INPUT 2048 /* samples per window, 0.17 s at 12 kHz */
#define WD_CLASSES 10
#define WD_BUF 3984   /* largest activation: 16 channels x 249 */

/* x: one window, mean removed and divided by the training scale (see
 * isense.cnn.prepare). Writes WD_CLASSES logits; returns the class index. */
int wdcnn_classify(const float *x, float *logits);

#endif
