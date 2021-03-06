#ifndef STINGER_DYNAMIC_BETWEENNESS_H_
#define STINGER_DYNAMIC_BETWEENNESS_H_

#include <stdint.h>
#include <unistd.h>
#include <stdbool.h>
#include <vector>
#include "stinger_net/stinger_alg.h"
#include "streaming_algorithm.h"
namespace gt {
  namespace stinger {
    class BetweennessCentrality : public IDynamicGraphAlgorithm
    {
    private:
        double * bc;
        int64_t * times_found;
        double * sample_bc;
        double weighting;
        uint8_t do_weighted;
        double old_weighting;

        std::vector<int64_t> vertices_to_sample;
    public:
        BetweennessCentrality(int64_t num_samples, double weighting, uint8_t do_weighted);
        ~BetweennessCentrality();

        void setSources(const std::vector<int64_t> &sources);

        // Overridden from IDynamicGraphAlgorithm
        std::string getName();
        int64_t getDataPerVertex();
        std::string getDataDescription();
        void onInit(stinger_registered_alg * alg);
        void onPre(stinger_registered_alg * alg);
        void onPost(stinger_registered_alg * alg);
    };
  }
}

#endif