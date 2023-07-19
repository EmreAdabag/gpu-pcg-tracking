
#include "interface.cuh"






int main(){

    struct pstorage<float> p;

    setupTracking_pcg<float>(&p);

    cleanupTracking_pcg<float>(&p);

    return 0;
}