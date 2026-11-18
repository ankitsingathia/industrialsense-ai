/* PC test driver: read windows from a file, write logits to another.
 *   in : int32 n, then n * WD_INPUT float32
 *   out: n * WD_CLASSES float32
 */
#include <stdint.h>
#include <stdio.h>

#include "wdcnn.h"

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s windows.bin logits.bin\n", argv[0]);
        return 2;
    }
    FILE *fi = fopen(argv[1], "rb");
    FILE *fo = fopen(argv[2], "wb");
    if (!fi || !fo) {
        perror("open");
        return 1;
    }
    int32_t n = 0;
    if (fread(&n, sizeof n, 1, fi) != 1) {
        return 1;
    }
    static float x[WD_INPUT];
    float logits[WD_CLASSES];
    for (int32_t i = 0; i < n; i++) {
        if (fread(x, sizeof(float), WD_INPUT, fi) != WD_INPUT) {
            return 1;
        }
        wdcnn_classify(x, logits);
        fwrite(logits, sizeof(float), WD_CLASSES, fo);
    }
    fclose(fi);
    fclose(fo);
    return 0;
}
