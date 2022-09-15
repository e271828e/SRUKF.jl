using Revise

include("test_srut.jl"); using .TestSRUT; test_srut()
include("test_stages.jl"); using .TestStages; test_stages()