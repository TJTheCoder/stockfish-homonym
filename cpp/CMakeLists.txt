cmake_minimum_required(VERSION 3.16)
project(stockfish_homonym_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(STOCK_PLATFORM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/market/market.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/market/order.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/market/orderbook.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/market/stock.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/market/trader.cpp"
)

add_executable(stockfish_platform_bridge
    src/bridge_main.cpp
    src/platform_execution_engine.cpp
    ${STOCK_PLATFORM_SOURCES}
)

target_include_directories(stockfish_platform_bridge PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(stockfish_platform_bridge PRIVATE pthread)
