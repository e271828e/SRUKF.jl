#not needed if we dev the package path in the docs environment
# push!(LOAD_PATH,"../src/")

#this requires adding Documenter either to the base environment or our local
#package docs environment
using Documenter

#this requires devving our package path from the package docs environment. at
#some point we should probably add the package from its GitHub URL instead
using SRUKF

makedocs(sitename="My Documentation")