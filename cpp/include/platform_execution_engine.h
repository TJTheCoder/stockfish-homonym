#ifndef STOCKFISH_HOMONYM_PLATFORM_EXECUTION_ENGINE_H
#define STOCKFISH_HOMONYM_PLATFORM_EXECUTION_ENGINE_H

#include <array>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../../stock-trading-platform/server/include/market.h"

namespace stockfish_homonym {

struct PlatformConfig {
    int target_inventory = 250;
    int horizon = 60;
    int warmup_steps = 20;
    int market_cap = 5000;
    double initial_balance = 1'000'000.0;
    double lambda_risk = 0.02;
    double lambda_urgency = 0.5;
};

struct StepResult {
    std::vector<double> observation;
    double reward = 0.0;
    bool terminated = false;
    bool truncated = false;
};

class PlatformExecutionEngine {
   public:
    enum class Regime { CALM = 0, NORMAL = 1, STRESSED = 2 };

    struct RegimeSpec {
        int brownian_updates_per_step;
        double risk_scale;
        double terminal_scale;
    };

    explicit PlatformExecutionEngine(PlatformConfig config);

    StepResult reset(int seed, bool calm_only);
    StepResult step(int action);

    bool episode_active() const;

   private:
    void initialize_market();
    void warmup_market();
    void advance_regime();
    void advance_prices();
    void select_target_stock();
    int execute_action(int action);
    std::vector<std::string> recommendation_ranking(int risk_tolerance) const;
    std::vector<double> build_observation() const;
    double current_price() const;
    double current_equity() const;
    double current_unrealized_pnl() const;
    int current_target_position() const;
    std::vector<double> last_target_returns(int window) const;
    double target_realized_volatility() const;
    double normalized_rank(
        const std::vector<std::string>& ranking,
        const std::string& symbol
    ) const;
    int symbol_index(const std::string& symbol) const;
    std::string transition_json(const StepResult& result) const;
    static std::vector<std::string> split(const std::string& text, const std::string& delim);

   public:
    std::string reset_json(int seed, bool calm_only);
    std::string step_json(int action);

   private:
    PlatformConfig config_;
    mutable std::mt19937 rng_;
    std::unique_ptr<Market> market_;
    std::vector<std::unique_ptr<Stock>> stocks_;
    std::unordered_map<std::string, Stock*> stocks_by_symbol_;
    std::vector<std::string> symbols_;
    std::vector<double> initial_prices_;
    std::unordered_map<std::string, int> remaining_liquidity_;
    std::vector<std::string> low_risk_ranking_;
    std::vector<std::string> medium_risk_ranking_;
    std::vector<std::string> high_risk_ranking_;
    std::string target_symbol_;
    int target_risk_tolerance_ = 2;
    int step_count_ = 0;
    int inventory_remaining_ = 0;
    int total_filled_ = 0;
    double arrival_price_ = 0.0;
    double shortfall_ = 0.0;
    double cash_spent_ = 0.0;
    double starting_balance_ = 0.0;
    bool calm_only_ = false;
    bool active_episode_ = false;
    Regime regime_ = Regime::NORMAL;
};

}  // namespace stockfish_homonym

#endif
