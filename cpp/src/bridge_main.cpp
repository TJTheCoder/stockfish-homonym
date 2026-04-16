#include "platform_execution_engine.h"

#include <cctype>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using stockfish_homonym::PlatformConfig;
using stockfish_homonym::PlatformExecutionEngine;

namespace {

PlatformConfig parse_config(int argc, char** argv) {
    PlatformConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for flag " + flag);
            }
            return argv[++i];
        };

        if (arg == "--target-inventory") {
            config.target_inventory = std::stoi(require_value(arg));
        } else if (arg == "--horizon") {
            config.horizon = std::stoi(require_value(arg));
        } else if (arg == "--warmup-steps") {
            config.warmup_steps = std::stoi(require_value(arg));
        } else if (arg == "--market-cap") {
            config.market_cap = std::stoi(require_value(arg));
        } else if (arg == "--initial-balance") {
            config.initial_balance = std::stod(require_value(arg));
        } else if (arg == "--lambda-risk") {
            config.lambda_risk = std::stod(require_value(arg));
        } else if (arg == "--lambda-urgency") {
            config.lambda_urgency = std::stod(require_value(arg));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return config;
}

std::vector<std::string> split_ws(const std::string& line) {
    std::vector<std::string> parts;
    std::string current;
    for (const char ch : line) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!current.empty()) {
                parts.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        parts.push_back(current);
    }
    return parts;
}

std::string error_json(const std::string& message) {
    return "{\"error\":\"" + message + "\"}";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        PlatformExecutionEngine engine(parse_config(argc, argv));
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }

            try {
                const std::vector<std::string> parts = split_ws(line);
                const std::string command = parts.front();
                if (command == "PING") {
                    std::cout << "{\"ok\":true}" << std::endl;
                } else if (command == "RESET") {
                    if (parts.size() != 3) {
                        throw std::runtime_error("RESET expects: RESET <seed> <calm_only>");
                    }
                    const int seed = std::stoi(parts[1]);
                    const bool calm_only = std::stoi(parts[2]) != 0;
                    std::cout << engine.reset_json(seed, calm_only) << std::endl;
                } else if (command == "STEP") {
                    if (parts.size() != 2) {
                        throw std::runtime_error("STEP expects: STEP <action>");
                    }
                    const int action = std::stoi(parts[1]);
                    std::cout << engine.step_json(action) << std::endl;
                } else if (command == "CLOSE") {
                    break;
                } else {
                    throw std::runtime_error("Unknown command: " + command);
                }
            } catch (const std::exception& e) {
                std::cout << error_json(e.what()) << std::endl;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << error_json(e.what()) << std::endl;
        return 1;
    }
}
