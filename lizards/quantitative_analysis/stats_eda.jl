using CSV, DataFrames, DataFramesMeta, StatsBase, RCall, Chain, CategoricalArrays, TexTables, GLM, RegressionTables
using Distributions, FreqTables, PrettyTables
using CairoMakie, AlgebraOfGraphics
using AlgebraOfGraphics: density

@rlibrary GGally
@rlibrary ggplot2
@rlibrary ggResidpanel



selfplay_dir = "/Users/ymh/Downloads/selfplay_vs_baseline_60trials_16.00.18"
pretrained_dir = "/Users/ymh/Downloads/pretrained_vs_baseline_60trials_17.00.28"
randtrained_dir = "/Users/ymh/Downloads/randtrained_vs_baseline_60trials_17.22.34"
matchup_dirs = [selfplay_dir, pretrained_dir, randtrained_dir]



df = DataFrame(CSV.File("/Users/ymh/Downloads/pretrained_vs_baseline_60trials_17.00.28/trial_49/AGENT_ATTACKS_TEMPORAL.csv"))


function get_avg_from_df(df)
  blue_df = df[:, [:blue_0, :blue_1, :blue_2, :blue_3, :blue_4, :blue_5]] 

  col_means = [mean(skipmissing(col)) for col in eachcol(blue_df)]
  avg_atk_per_agent_for_blue_on_trial = mean(col_means)

  return avg_atk_per_agent_for_blue_on_trial
end


function get_avges_from_matchup(matchup_dir)
  atk_filepaths = [joinpath(root, f) for (root, _, files) in walkdir(matchup_dir) for f in files if startswith(f, "AGENT_ATTACKS_TEMPORAL")]

  avges = map(atk_filepaths) do csv_filepath
      df = DataFrame(CSV.File(csv_filepath))
      get_avg_from_df(df)
  end
  return avges
end

selfplay_avges = get_avges_from_matchup(selfplay_dir) 
sp_avg = selfplay_avges |> mean

pretrained_avg = get_avges_from_matchup(pretrained_dir) |> mean

get_avges_from_matchup(randtrained_dir) |> mean







# make win-loss stats



function get_winning_team_from_df(df)
  ## Actually can just use HP from last row!
  if !ismissing(last(df).hp_red) && !ismissing(last(df).hp_red)
    hp_red = last(df).hp_red
    hp_blue = last(df).hp_blue 
  else
    penult_row = df[nrow(df)-1, :]
    hp_red = penult_row.hp_red
    hp_blue = penult_row.hp_blue
  end

  println(hp_red)
  println(hp_blue)

  if round(hp_red) - round(hp_blue) >= 2
    winning_team = "red" 
  elseif round(hp_blue) - round(hp_red) >= 2
    winning_team = "blue"
  else
    winning_team = "tied"
  end

  return winning_team
end



function get_wins_from_matchup(matchup_dir)
  timeline_filepaths = [joinpath(root, f) for (root, _, files) in walkdir(matchup_dir) for f in files if startswith(f, "ABSOLUTE_TIMELINE")]

  trial_info_list_for_matchup = map(timeline_filepaths) do csv_filepath
      df = DataFrame(CSV.File(csv_filepath))
      get_winning_team_from_df(df)
    end

  trial_info_list_for_matchup |> categorical
end


pretrained_wins = get_wins_from_matchup(pretrained_dir) 
freqtable(pretrained_wins)


randtrained_wins = get_wins_from_matchup(randtrained_dir) 
freqtable(randtrained_wins)


selfplay_wins = get_wins_from_matchup(selfplay_dir) 
freqtable(selfplay_wins)


#   return trial_info_list_for_matchup

#   # trial_df_for_matchup = DataFrame(trial_idx = [], who_won=String[])
#   # for row in trial_info_list_for_matchup
#   #     push!(trial_df_for_matchup, row[1])
#   # end

#   # return trial_df_for_matchup
# end

freqtable(pretrained_wins, :who_won)
freqtable(randtrained_wins, :who_won)

describe(pretrained_wins)



# Start by just getting win / loss for one trial df
splitdir(timeline_filepaths[50])[1] |> splitdir |> x->x[2] |> s->split(s, "_") |> x->x[2]



test2_df = DataFrame(CSV.File(t2))



