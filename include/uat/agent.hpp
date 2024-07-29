//! \file agent.hpp
//! \brief Defines the agent class and related types.

#ifndef UAT_AGENT_HPP
#define UAT_AGENT_HPP

#include <uat/type.hpp>
#include <uat/permit.hpp>

#include <stdexcept>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

namespace uat
{

//! Represents the public values in a trade of a permit.
struct trade_value_t {
  value_t min_value; //!< The minimum value the owner asked for the permit.
  value_t highest_bid; //!< The highest bid for the permit.
};

namespace permit_public_status
{

//! Represents the public status of a permit that is not available for trading.
struct unavailable
{};

//! \brief Represents the public status of a permit that is available for trading.
//!
//! - `min_value`: the minimum value that can be offered for the permit.
//! - `trades`: a function that lazily returns the history of trades for the permit.
//!             Each element contains the minimum value and the highest bid.
struct available
{
  value_t min_value;
  std::function<const std::vector<trade_value_t>&()> trades;
};

//! Represents the public status of a permit that is owned by the agent.
struct owned
{};
} // namespace permit_public_status

//! \brief Variant that represents the possible public status of a permit.
//!
//! - `unavailable`: the permit is not available for trading.
//! - `available`: the permit is available for trading. The `min_value` field
//!                represents the minimum value that can be offered for the permit.
//!                The `trades` field is a function that lazily returns the history
//!                of trades for the permit, where each elements contains the minimum
//!                value and the highest bid.
//! - `owned`: the permit is owned by the agent.
//!
//! \relates uat::permit_public_status_t
using permit_public_status_t =
  std::variant<permit_public_status::unavailable, permit_public_status::available, permit_public_status::owned>;

// TODO: is it possible to use function_ref?
using bid_fn = std::function<bool(region_view, uint_t, value_t)>;
using ask_fn = std::function<bool(region_view, uint_t, value_t)>;
using permit_public_status_fn = std::function<permit_public_status_t(region_view, uint_t)>;

template <region R>
struct agent_for
{
  using region_type = R;

  virtual auto bid_phase(uint_t, bid_fn, permit_public_status_fn, int) -> void {}
  virtual auto ask_phase(uint_t, ask_fn, permit_public_status_fn, int) -> void {}

  virtual auto on_bought(const R&, uint_t, value_t) -> void {}
  virtual auto on_sold(const R&, uint_t, value_t) -> void {}

  virtual auto stop(uint_t, int) -> bool = 0;
};

template <typename T>
concept compatible_agent = std::movable<T> &&
  std::derived_from<T, agent_for<typename T::region_type>>;

//! \brief A type-erased class that represents an agent in the simulation.
class agent
{
  //! \private
  class agent_interface
  {
  public:
    virtual ~agent_interface() = default;

    virtual auto bid_phase(uint_t, bid_fn, permit_public_status_fn, int) -> void = 0;
    virtual auto ask_phase(uint_t, ask_fn, permit_public_status_fn, int) -> void = 0;

    virtual auto on_bought(region_view, uint_t, value_t) -> void = 0;
    virtual auto on_sold(region_view, uint_t, value_t) -> void = 0;

    virtual auto stop(uint_t, int) -> bool = 0;
  };

  //! \private
  template <compatible_agent Agent> class agent_model : public agent_interface
  {
  public:
    agent_model(Agent agent) : agent_(std::move(agent)) {}
    virtual ~agent_model() = default;

    auto bid_phase(uint_t t, bid_fn b, permit_public_status_fn i, int seed) -> void override
    {
      agent_.bid_phase(t, std::move(b), std::move(i), seed);
    }

    auto ask_phase(uint_t t, ask_fn a, permit_public_status_fn i,
                   int seed) -> void override
    {
      agent_.ask_phase(t, std::move(a), std::move(i), seed);
    }

    auto on_bought(region_view s, uint_t t, value_t v) -> void override
    {
      agent_.on_bought(s.downcast<typename Agent::region_type>(), t, v);
    }

    auto on_sold(region_view s, uint_t t, value_t v) -> void override
    {
      agent_.on_sold(s.downcast<typename Agent::region_type>(), t, v);
    }

    auto stop(uint_t t, int seed) -> bool override { return agent_.stop(t, seed); }

  private:
    Agent agent_;
  };

public:
  //! Constructs a type-erased agent from an object of type Agent that satisfy at least:
  //!
  //! - `Agent::stop(uint_t, int) -> convertible_to<bool>`
  //!
  //! The Agent object should be copyable.
  template <compatible_agent Agent> agent(Agent a) : interface_(new agent_model<Agent>(std::move(a)))
  {
  }

  agent() = delete;

  agent(const agent& other) = delete;
  agent(agent&& other) noexcept = default;

  auto operator=(const agent& other) -> agent& = delete;
  auto operator=(agent&& other) noexcept -> agent& = default;

  //! Behavior of the agent during the bid phase.
  //!
  //! \param t The current time step.
  //! \param bid A function that allows the agent to bid for a permit.
  //! \param status A function that returns the public status of a permit.
  //! \param seed A random seed.
  //!
  //! The function `bid` and `status` receive a `region` and a `uint_t` as arguments
  //! representing the location and time of the permit.  The third argument of `bid`
  //! is the value the agent is willing to bid for the permit. The function `bid` returns
  //! true if the bidding was successful, and false otherwise.  The function `status`
  //! returns the public status of the permit with type `permit_public_status_t`.
  //!
  //! \note This function is only called if the agent has a `bid_phase` member function
  //!       with **compatible signature**.  Differently from direct inheritance, the
  //!       compiler will not check if the function signature is compatible. Use
  //!       `static_assert(is_detected_exact_v<void, mb_bid_phase_t, Agent>, "message")`
  //!       to ensure the function signature is correct.
  auto bid_phase(uint_t t, bid_fn bid, permit_public_status_fn status, int seed) -> void;

  //! Behavior of the agent during the ask phase.
  //!
  //! \param t The current time step.
  //! \param ask A function that allows the agent to ask for a permit.
  //! \param status A function that returns the public status of a permit.
  //! \param seed A random seed.
  //!
  //! The function `ask` and `status` receive a `region` and a `uint_t` as arguments
  //! representing the location and time of the permit.  The third argument of `ask`
  //! is the value the agent is willing to ask for the permit. The function `ask` returns
  //! true if the asking was successful, and false otherwise.  The function `status`
  //! returns the public status of the permit with type `permit_public_status_t`.
  //!
  //! \note This function is only called if the agent has a `ask_phase` member function
  //!       with **compatible signature**.  Differently from direct inheritance, the
  //!       compiler will not check if the function signature is compatible. Use
  //!       `static_assert(is_detected_exact_v<void, mb_ask_phase_t, Agent>, "message")`
  //!       to ensure the function signature is correct.
  auto ask_phase(uint_t t, ask_fn ask, permit_public_status_fn status, int seed) -> void;

  //! Callback function called when the agent successfully buys a permit.
  //!
  //! \param region The region of the permit.
  //! \param time The time of the permit.
  //! \param value The value paid for the permit.
  //!
  //! \note This function is only called if the agent has a `on_bought` member function
  //!       with **compatible signature**.  Differently from direct inheritance, the
  //!       compiler will not check if the function signature is compatible. Use
  //!       `static_assert(is_detected_exact_v<void, mb_on_bought_t, Agent>, "message")`
  //!       to ensure the function signature is correct.
  auto on_bought(region_view region, uint_t time, value_t value) -> void;

  //! Callback function called when the agent successfully sells a permit.
  //!
  //! \param region The region of the permit.
  //! \param time The time of the permit.
  //! \param value The value received for the permit.
  //!
  //! \note This function is only called if the agent has a `on_sold` member function
  //!       with **compatible signature**.  Differently from direct inheritance, the
  //!       compiler will not check if the function signature is compatible. Use
  //!       `static_assert(is_detected_exact_v<void, mb_on_sold_t, Agent>, "message")`
  //!       to ensure the function signature is correct.
  auto on_sold(region_view region, uint_t time, value_t value) -> void;

  //! Controls when the agent should stop.
  //!
  //! Once this function returns true, the agent will be removed from the simulation.
  //!
  //! \param time The current time step.
  //! \param seed A random seed.
  auto stop(uint_t time, int seed) -> bool;

private:
  std::unique_ptr<agent_interface> interface_;
};

} // namespace uat

#endif // UAT_AGENT_HPP
