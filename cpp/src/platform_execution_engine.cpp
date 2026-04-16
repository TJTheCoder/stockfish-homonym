#include "platform_execution_engine.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace stockfish_homonym {

namespace {

constexpr int kTraderId = 4242;
constexpr int kObservationSize = 58;

PlatformExecutionEngine::RegimeSpec regime_spec(PlatformExecutionEngine::Regime regime) {
    switch (regime) {
        case PlatformExecutionEngine::Regime::CALM:
            return {.brownian_updates_per_step = 1, .risk_scale = 0.6, .terminal_scale = 1.0};
        case PlatformExecutionEngine::Regime::NORMAL:
            return {.brownian_updates_per_step = 2, .risk_scale = 1.0, .terminal_scale = 1.4};
        case PlatformExecutionEngine::Regime::STRESSED:
            return {.brownian_updates_per_step = 4, .risk_scale = 1.8, .terminal_scale = 2.2};
    }
    return {.brownian_updates_per_step = 2, .risk_scale = 1.0, .terminal_scale = 1.4};
}

double clip(double value, double low, double high) {
    return std::max(low, std::min(high, value));
}

std::string bool_json(bool value) {
    return value ? "true" : "false";
}

}  // namespace

PlatformExecutionEngine::PlatformExecutionEngine(PlatformConfig config)
    : config_(config), rng_(0) {}

bool PlatformExecutionEngine::episode_active() const {
    return active_episode_;
}

void PlatformExecutionEngine::initialize_market() {
    market_ = std::make_unique<Market>();
    stocks_.clear();
    stocks_by_symbol_.clear();
    symbols_.clear();
    initial_prices_.clear();
    remaining_liquidity_.clear();

    struct StockSeed {
        const char* symbol;
        const char* name;
        double initial_price;
    };

    const std::array<StockSeed, 10> stock_seeds = {{
        {"AAPL", "Apple Inc.", 259.28},
        {"TSLA", "Tesla Inc.", 454.13},
        {"AMZN", "Amazon.com Inc.", 227.62},
        {"GOOG", "Alphabet Inc. Class C", 197.48},
        {"MSFT", "Microsoft Corp.", 438.94},
        {"NVDA", "NVIDIA Corp.", 137.09},
        {"GME", "GameStop Corp.", 32.20},
        {"INTC", "Intel Corp.", 20.32},
        {"DIS", "Walt Disney Co.", 111.55},
        {"PYPL", "PayPal Holdings Inc.", 86.86},
    }};

    stocks_.reserve(stock_seeds.size());
    symbols_.reserve(stock_seeds.size());
    initial_prices_.reserve(stock_seeds.size());
    for (const auto& seed : stock_seeds) {
        auto stock = std::make_unique<Stock>(seed.symbol, seed.name, seed.initial_price);
        symbols_.push_back(seed.symbol);
        initial_prices_.push_back(seed.initial_price);
        remaining_liquidity_[seed.symbol] = config_.market_cap;
        stocks_by_symbol_[seed.symbol] = stock.get();
        market_->add_stock(*stock, config_.market_cap);
        stocks_.push_back(std::move(stock));
    }

    market_->add_trader(kTraderId, static_cast<float>(config_.initial_balance));
    starting_balance_ = config_.initial_balance;
}

void PlatformExecutionEngine::warmup_market() {
    for (int i = 0; i < config_.warmup_steps; ++i) {
        advance_regime();
        advance_prices();
    }
}

void PlatformExecutionEngine::advance_regime() {
    if (calm_only_) {
        regime_ = Regime::CALM;
        return;
    }

    static const std::array<std::array<double, 3>, 3> transitions = {{
        {0.92, 0.07, 0.01},
        {0.08, 0.84, 0.08},
        {0.02, 0.10, 0.88},
    }};
    const int current = static_cast<int>(regime_);
    std::discrete_distribution<int> distribution(
        transitions[current].begin(),
        transitions[current].end()
    );
    regime_ = static_cast<Regime>(distribution(rng_));
}

void PlatformExecutionEngine::advance_prices() {
    const RegimeSpec spec = regime_spec(regime_);
    for (const auto& stock : stocks_) {
        for (int i = 0; i < spec.brownian_updates_per_step; ++i) {
            stock->brownian_motion();
        }
    }
}

std::vector<std::string> PlatformExecutionEngine::recommendation_ranking(int risk_tolerance) const {
    const std::string request =
        "RECOMMEND:" + std::to_string(risk_tolerance) + ":" + std::to_string(symbols_.size());
    const std::string response = market_->recommend_stocks(request);
    const std::string prefix = "Recommended stocks: ";
    if (response.rfind(prefix, 0) != 0) {
        throw std::runtime_error("Unexpected recommendation response: " + response);
    }
    return split(response.substr(prefix.size()), ", ");
}

void PlatformExecutionEngine::select_target_stock() {
    std::uniform_int_distribution<int> risk_distribution(1, 3);
    target_risk_tolerance_ = risk_distribution(rng_);
    low_risk_ranking_ = recommendation_ranking(1);
    medium_risk_ranking_ = recommendation_ranking(2);
    high_risk_ranking_ = recommendation_ranking(3);

    if (target_risk_tolerance_ == 1) {
        target_symbol_ = low_risk_ranking_.front();
    } else if (target_risk_tolerance_ == 2) {
        target_symbol_ = medium_risk_ranking_.front();
    } else {
        target_symbol_ = high_risk_ranking_.front();
    }

    arrival_price_ = current_price();
}

StepResult PlatformExecutionEngine::reset(int seed, bool calm_only) {
    rng_.seed(seed);
    std::srand(seed);
    calm_only_ = calm_only;
    regime_ = calm_only ? Regime::CALM : Regime::NORMAL;
    step_count_ = 0;
    inventory_remaining_ = config_.target_inventory;
    total_filled_ = 0;
    shortfall_ = 0.0;
    cash_spent_ = 0.0;
    active_episode_ = true;

    initialize_market();
    warmup_market();
    select_target_stock();

    StepResult result;
    result.observation = build_observation();
    return result;
}

int PlatformExecutionEngine::execute_action(int action) {
    if (action < 0 || action > 5) {
        throw std::out_of_range("Action must be in [0, 5]");
    }
    if (inventory_remaining_ <= 0) {
        return 0;
    }

    const int base_child = std::max(1, config_.target_inventory / config_.horizon);
    int quantity = 0;
    switch (action) {
        case 0:
            quantity = 0;
            break;
        case 1:
            quantity = base_child;
            break;
        case 2:
            quantity = base_child * 2;
            break;
        case 3:
            quantity = base_child * 4;
            break;
        case 4:
            quantity = base_child * 8;
            break;
        case 5:
            quantity = inventory_remaining_;
            break;
    }
    quantity = std::min(quantity, inventory_remaining_);
    if (quantity == 0) {
        return 0;
    }

    Trader& trader = market_->get_trader(kTraderId);
    const double fill_price = current_price();
    const int affordable = static_cast<int>(std::floor(trader.get_balance() / fill_price));
    const int available = remaining_liquidity_.at(target_symbol_);
    const int filled = std::max(0, std::min(quantity, std::min(available, affordable)));
    if (filled > 0) {
        trader.update_balance(static_cast<float>(-fill_price * filled));
        trader.update_quantity(target_symbol_, filled);
        remaining_liquidity_[target_symbol_] -= filled;
        inventory_remaining_ -= filled;
        total_filled_ += filled;
        cash_spent_ += fill_price * filled;
        shortfall_ += (fill_price - arrival_price_) * filled;
    }
    return filled;
}

double PlatformExecutionEngine::current_price() const {
    return stocks_by_symbol_.at(target_symbol_)->get_price();
}

double PlatformExecutionEngine::current_equity() const {
    Trader& trader = market_->get_trader(kTraderId);
    const double cash = trader.get_balance();
    const double inventory_value = current_target_position() * current_price();
    return cash + inventory_value;
}

double PlatformExecutionEngine::current_unrealized_pnl() const {
    const int held = current_target_position();
    if (held <= 0) {
        return 0.0;
    }
    return held * current_price() - cash_spent_;
}

int PlatformExecutionEngine::current_target_position() const {
    return market_->get_trader(kTraderId).get_quantity(target_symbol_);
}

std::vector<double> PlatformExecutionEngine::last_target_returns(int window) const {
    const std::vector<float> history = stocks_by_symbol_.at(target_symbol_)->get_price_history();
    std::vector<double> out(window, 0.0);
    if (history.size() < 2) {
        return out;
    }

    int write = window - 1;
    for (int i = static_cast<int>(history.size()) - 1; i > 0 && write >= 0; --i, --write) {
        const double prev = static_cast<double>(history[static_cast<size_t>(i - 1)]);
        const double cur = static_cast<double>(history[static_cast<size_t>(i)]);
        if (prev <= 0.0 || cur <= 0.0) {
            out[static_cast<size_t>(write)] = 0.0;
            continue;
        }
        out[static_cast<size_t>(write)] = clip(std::log(cur / prev) / 0.02, -1.0, 1.0);
    }
    return out;
}

double PlatformExecutionEngine::target_realized_volatility() const {
    const std::vector<double> returns = last_target_returns(10);
    double mean = 0.0;
    for (const double value : returns) {
        mean += value;
    }
    mean /= static_cast<double>(returns.size());

    double variance = 0.0;
    for (const double value : returns) {
        const double centered = value - mean;
        variance += centered * centered;
    }
    variance /= static_cast<double>(returns.size());
    return std::sqrt(variance);
}

double PlatformExecutionEngine::normalized_rank(
    const std::vector<std::string>& ranking,
    const std::string& symbol
) const {
    for (size_t i = 0; i < ranking.size(); ++i) {
        if (ranking[i] == symbol) {
            return ranking.size() <= 1
                ? 0.0
                : static_cast<double>(i) / static_cast<double>(ranking.size() - 1);
        }
    }
    return 1.0;
}

int PlatformExecutionEngine::symbol_index(const std::string& symbol) const {
    for (size_t i = 0; i < symbols_.size(); ++i) {
        if (symbols_[i] == symbol) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::vector<double> PlatformExecutionEngine::build_observation() const {
    std::vector<double> obs(kObservationSize, 0.0);
    const double price = current_price();
    const double inv_frac = static_cast<double>(inventory_remaining_) / config_.target_inventory;
    const double time_remaining =
        static_cast<double>(config_.horizon - step_count_) / std::max(1, config_.horizon);
    const double urgency = clip(inv_frac / std::max(1e-6, time_remaining), 0.0, 5.0) / 5.0;
    const auto target_returns = last_target_returns(10);

    obs[0] = clip(time_remaining, 0.0, 1.0);
    obs[1] = clip(inv_frac, 0.0, 1.0);
    obs[2] = clip(shortfall_ / (arrival_price_ * config_.target_inventory + 1e-6), -1.0, 1.0);
    obs[3] = clip(cash_spent_ / (arrival_price_ * config_.target_inventory + 1e-6), 0.0, 5.0);
    obs[4] = static_cast<double>(target_risk_tolerance_ - 1) / 2.0;
    obs[5] = clip((price / arrival_price_ - 1.0) * 20.0, -1.0, 1.0);
    obs[6] = clip(target_realized_volatility(), 0.0, 1.0);
    obs[7] = target_returns[8];
    obs[8] = target_returns[9];
    obs[9] = normalized_rank(low_risk_ranking_, target_symbol_);
    obs[10] = normalized_rank(medium_risk_ranking_, target_symbol_);
    obs[11] = normalized_rank(high_risk_ranking_, target_symbol_);
    obs[12] = low_risk_ranking_.front() == target_symbol_ ? 1.0 : 0.0;
    obs[13] = medium_risk_ranking_.front() == target_symbol_ ? 1.0 : 0.0;
    obs[14] = high_risk_ranking_.front() == target_symbol_ ? 1.0 : 0.0;
    obs[15] = clip(current_equity() / starting_balance_, 0.0, 2.0);
    obs[16] = clip(current_unrealized_pnl() / (arrival_price_ * config_.target_inventory + 1e-6), -1.0, 1.0);
    obs[17] = urgency;

    for (size_t i = 0; i < 10; ++i) {
        obs[18 + i] = target_returns[i];
    }

    for (size_t i = 0; i < symbols_.size(); ++i) {
        const double cur = stocks_[i]->get_price();
        const double init = initial_prices_[i];
        obs[28 + i] = clip((cur / init - 1.0) * 10.0, -1.0, 1.0);
    }

    for (size_t i = 0; i < symbols_.size(); ++i) {
        const double sigma = clip(stocks_[i]->standard_deviation() / 20.0, 0.0, 1.0);
        obs[38 + i] = sigma;
    }

    const int target_idx = symbol_index(target_symbol_);
    if (target_idx >= 0) {
        obs[48 + static_cast<size_t>(target_idx)] = 1.0;
    }

    return obs;
}

StepResult PlatformExecutionEngine::step(int action) {
    if (!active_episode_) {
        throw std::runtime_error("Episode is inactive. Call reset() before step().");
    }

    advance_regime();
    const RegimeSpec spec = regime_spec(regime_);
    const int filled = execute_action(action);
    const double price_before_move = current_price();

    advance_prices();
    low_risk_ranking_ = recommendation_ranking(1);
    medium_risk_ranking_ = recommendation_ranking(2);
    high_risk_ranking_ = recommendation_ranking(3);
    step_count_ += 1;

    double reward = 0.0;
    if (filled > 0) {
        reward -= (price_before_move - arrival_price_) * filled;
    }
    const double inv_frac = static_cast<double>(inventory_remaining_) / config_.target_inventory;
    reward -= config_.lambda_risk * inv_frac * inv_frac * spec.risk_scale;

    bool terminated = inventory_remaining_ <= 0;
    bool truncated = step_count_ >= config_.horizon && !terminated;

    StepResult result;
    if (truncated && inventory_remaining_ > 0) {
        const double terminal_penalty =
            config_.lambda_urgency
            * spec.terminal_scale
            * inv_frac
            * clip(current_price() / arrival_price_, 0.5, 2.0);
        reward -= terminal_penalty;
    }
    result.reward = reward;
    result.terminated = terminated;
    result.truncated = truncated;
    result.observation = build_observation();
    if (terminated || truncated) {
        active_episode_ = false;
    }
    return result;
}

std::vector<std::string> PlatformExecutionEngine::split(const std::string& text, const std::string& delim) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        const size_t pos = text.find(delim, start);
        if (pos == std::string::npos) {
            out.push_back(text.substr(start));
            return out;
        }
        out.push_back(text.substr(start, pos - start));
        start = pos + delim.size();
    }
}

std::string PlatformExecutionEngine::transition_json(const StepResult& result) const {
    std::ostringstream oss;
    oss << std::setprecision(8);
    oss << "{";
    oss << "\"obs\":[";
    for (size_t i = 0; i < result.observation.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << result.observation[i];
    }
    oss << "],";
    oss << "\"reward\":" << result.reward << ",";
    oss << "\"terminated\":" << bool_json(result.terminated) << ",";
    oss << "\"truncated\":" << bool_json(result.truncated) << ",";
    oss << "\"info\":{";
    oss << "\"step\":" << step_count_ << ",";
    oss << "\"inventory_remaining\":" << inventory_remaining_ << ",";
    oss << "\"actual_fills\":" << total_filled_ << ",";
    oss << "\"shortfall_so_far\":" << shortfall_ << ",";
    oss << "\"target_risk_tolerance\":" << target_risk_tolerance_ << ",";
    oss << "\"target_symbol_index\":" << symbol_index(target_symbol_) << ",";
    oss << "\"target_symbol\":\"" << target_symbol_ << "\",";
    oss << "\"arrival_price\":" << arrival_price_ << ",";
    oss << "\"current_price\":" << current_price() << ",";
    oss << "\"regime_id\":" << static_cast<int>(regime_) << ",";
    oss << "\"equity\":" << current_equity();
    oss << "}";
    oss << "}";
    return oss.str();
}

std::string PlatformExecutionEngine::reset_json(int seed, bool calm_only) {
    return transition_json(reset(seed, calm_only));
}

std::string PlatformExecutionEngine::step_json(int action) {
    return transition_json(step(action));
}

}  // namespace stockfish_homonym
