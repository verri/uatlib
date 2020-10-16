#include <uat/simulation.hpp>

#include <deque>
#include <unordered_map>
#include <functional>
#include <variant>
#include <random>
#include <limits>

#include <jules/core/ranges.hpp>
#include <cool/indices.hpp>
#include <cool/compose.hpp>
#include <fmt/format.h>

namespace uat
{


} // namespace uat

namespace uat
{

auto simulate(factory_t factory, airspace space, int seed, const simulation_opts_t& opts) -> void
{
  std::mt19937 rnd(seed);

  std::vector<agent> agents;
  std::vector<id_t> active, keep_active, to_finished;

  uint_t t0 = 0;

  std::deque<std::unordered_map<tslot, private_status>> data;

  private_status ool = out_of_limits{};
  auto book = [&t0, &data, &ool, &opts](const slot& slot, uint_t t) mutable -> private_status& {
    assert(t >= t0);
    if (opts.time_window && t > t0 + *opts.time_window)
      return ool;
    while (t - t0 >= data.size())
      data.emplace_back();
    return data[t - t0][{slot, t}];
  };

  auto public_access = [&book](auto id) {
    return [id = id, &book](const slot& s, uint_t t) -> slot_status {
      return std::visit(cool::compose{
        [](out_of_limits) -> slot_status { return unavailable{}; },
        [&](used status) -> slot_status { return status.owner == id ? slot_status{owned{}} : unavailable{}; },
        [&](onsale status) -> slot_status { return status.owner == id ? slot_status{unavailable{}} : available{status.min_value}; }
      }, book(s, t));
    };
  };

  const auto stop = [&] {
    return std::visit(cool::compose{
      [&](no_agents_t) { return active.size() == 0;  },
      [&](time_threshold_t th) { return t0 > th.t; },
    }, opts.stop_criteria);
  };

  do
  {
    if (opts.status_callback)
      opts.status_callback(t0, space, [&book](const slot& slot, uint_t t) -> private_status { return book(slot, t); });

    {
      using namespace jules::ranges;
      auto new_agents = factory(t0, space, rnd());
      copy(cool::indices(agents.size(), agents.size() + new_agents.size()), back_inserter(active));
      move(new_agents, back_inserter(agents));
    }

    keep_active.clear();
    keep_active.reserve(active.size());

    to_finished.clear();
    to_finished.reserve(active.size());

    {
      std::vector<std::tuple<slot, uint_t>> bids;
      for (const auto id : active)
      {
        const auto r = agents[id].act(t0, [&](const slot& s, uint_t t, value_t v) -> bool {
          if (t < t0) return false;
          return std::visit(cool::compose{
            [](out_of_limits) { return false; },
            [](used) { return false; },
            [&](onsale& status) {
              if (v > status.min_value && v > status.highest_bid)
              {
                if (status.highest_bidder == no_owner)
                  bids.emplace_back(s, t);

                status.highest_bidder = id;
                status.highest_bid = v;
              }
              return true;
            }
          }, book(s, t));
        }, public_access(id), rnd());

        if (r)
          keep_active.push_back(id);
        else
          to_finished.push_back(id);
      }

      for (const auto& [s, t] : bids)
      {
        const auto status = std::get<onsale>(book(s, t));
        if (opts.trade_callback)
          opts.trade_callback({t0, status.owner, status.highest_bidder, s, t, status.highest_bid});

        agents[status.highest_bidder].on_bought(s, t, status.highest_bid);
        if (status.owner != no_owner)
          agents[status.owner].on_sold(s, t, status.highest_bid);

        book(s, t) = used{status.highest_bidder};
      }
    }

    {
      std::vector<std::tuple<slot, uint_t, uint_t, value_t>> asks;
      for (const auto id : active)
      {
        agents[id].after_auction(t0, [&](const slot& s, uint_t t, value_t v) -> bool {
          if (t < t0) return false;
          return std::visit(cool::compose{
            [](out_of_limits) { return false; },
            [](onsale) { return false; },
            [&](used& status) {
              if (status.owner != id)
                return false;
              asks.emplace_back(s, t, id, v);
              return true;
            }
          }, book(s, t));
        }, public_access(id), rnd());
      }

      for (const auto& [s, t, id, v] : asks)
        book(s, t) = onsale{ .owner = id, .min_value = v };
    }

    for (const auto id : to_finished)
      agents[id].on_finished(id, t0);

    std::swap(active, keep_active);
    data.pop_front();
    ++t0;
  } while (!stop());
}

} // namespace uat
