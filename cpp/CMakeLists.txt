cmake_minimum_required(VERSION 3.16)
project(stockfish_homonym_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(STOCK_PLATFORM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vendor/stock_platform")

set(STOCK_PLATFORM_SOURCES
    "${STOCK_PLATFORM_ROOT}/src/market.cpp"
    "${STOCK_PLATFORM_ROOT}/src/order.cpp"
    "${STOCK_PLATFORM_ROOT}/src/orderbook.cpp"
    "${STOCK_PLATFORM_ROOT}/src/stock.cpp"
    "${STOCK_PLATFORM_ROOT}/src/trader.cpp"
)

add_executable(stockfish_platform_bridge
    src/bridge_main.cpp
    src/platform_execution_engine.cpp
    ${STOCK_PLATFORM_SOURCES}
)

target_include_directories(stockfish_platform_bridge PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${STOCK_PLATFORM_ROOT}/include"
)

target_link_libraries(stockfish_platform_bridge PRIVATE pthread)
