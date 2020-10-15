

function dependencies_violated(statements, dependencies, loops, loopdep)
    any(dep -> dependency_violated(statements, dep, loops, loopdep), dependencies)
end
function dependency_violated()
    
end

