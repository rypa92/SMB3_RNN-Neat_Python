-- Initial fitness score to 0
fitness = 0
max_fitness = 0
framecount = 0
prev_pos = {24, -128}
cur_pos = {24, -128}
max_pos = {24, -128}
screen_num = 0
game_score = 0
finish = false
game_time = 300

-- Work some magic on getting the location of mario
-- Used in calculating fitness
function calculate_fitness()

    if finish == true or framecount == 300 then
        fitness = fitness + game_time
        emu.pause()
    end

    -- Check if mario finished the level using the address in RAM for the cards you get
    if memory.readbyte(0x7D9C) ~= 0 then
        -- If so, we reward by adding 125000 to the fitness
        fitness = fitness + 125000
        finish = true
    end

    -- Update the current position of mario from RAM
    cur_pos = {memory.readbyte(0x0090) + (memory.readbyte(0x0075) * 256), memory.readbyte(0x00A2)}

    -- Check if we are standing in the same position as the previous frame
    if cur_pos ~= prev_pos then
        if cur_pos[1] > prev_pos[1] then
            fitness = fitness + 2
        end
        if cur_pos[1] > max_pos[1] then
            max_pos[1] = cur_pos[1]
        end
        if cur_pos[2] > max_pos[2] then
            max_pos[2] = cur_pos[2]
        end
        prev_pos = cur_pos
    end

    -- Check if the game_score is higher than the last frame
    if memory.readbyte(0x0715) > game_score then
        -- Add the difference of the score to the fitness
        fitness = fitness + (memory.readbyte(0x0715) - game_score)
        -- set game_score to the current score in game
        game_score = memory.readbyte(0x0715)
    end

    -- Check if we die
    if memory.readbyte(0x0736) ~= 4 then
        -- if we do, set finish to true
        finish = true
    end

    -- Check if the time in game is at least higher than 0
    if ((memory.readbyte(0x05EE) * 100) + (memory.readbyte(0x05EF) * 10) + memory.readbyte(0x05F0)) ~= 0 then
        -- if so, keep track of the game_time
        game_time = ((memory.readbyte(0x05EE) * 100) + (memory.readbyte(0x05EF) * 10) + memory.readbyte(0x05F0))
    end

    if fitness > max_fitness then
        max_fitness = fitness
        framecount = 0
    else
        framecount = framecount + 1
    end
end

-- Main loop while the emulator is running
while true do
    -- calculate the fitness score for this frame
	calculate_fitness()
    -- Output the current fitness and the Absolute pixel (x-axis only) that mario is at
    gui.text(10, 10, "Fitness: " .. fitness, "white", "black")
    gui.text(10, 20, "Position: [" .. cur_pos[1] .. ", " .. cur_pos[2] .. "]", "white", "black")
    gui.text(10, 30, "Frame Count: " .. framecount, "white", "black")
    -- Advance the frame of the emulator by 1
	emu.frameadvance()
end