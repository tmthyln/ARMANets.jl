push!(LOAD_PATH, "../src/")

using Documenter, ARMANets

makedocs(
    sitename="ARMANets.jl Documentation",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true"
    ),
    modules=[ARMANets],
    pages=[
        "Home" => "index.md"
    ]
)

deploydocs(
    repo = "github.com/tmthyln/ARMANets.jl.git",
    devbranch = "master",
    devurl="latest"
    )
