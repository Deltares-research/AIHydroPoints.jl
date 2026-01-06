# test_abstract_series.jl
#

function test_abstract_series()
    # Test find_location_index
    location_names = ["Station A", "Station B", "Station C"]
    index = find_location_index("Station A", location_names)
    @test index == 1

    index = find_location_index("Station D", location_names)
    @test index == -1

    # Test find_location_indices
    location_selection = ["Station A", "Station C", "Station D"]
    indices = find_location_indices(location_selection, location_names)
    @test indices == [1, 3, -1]
end

# Run the test
test_abstract_series()