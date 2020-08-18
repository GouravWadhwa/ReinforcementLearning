import numpy as np

class Tiles () :
    def __init__ (
        self, 
        min_value_x, 
        max_value_x, 
        min_value_y, 
        max_value_y, 
        num_tiles = 4, 
        boxes_per_tile_x = 4, 
        boxes_per_tile_y = 4
    ) :
        self.min_value_x = min_value_x
        self.max_value_x = max_value_x
        self.min_value_y = min_value_y
        self.max_value_y = max_value_y

        self.num_tiles = num_tiles
        self.boxes_per_tile_x = boxes_per_tile_x
        self.boxes_per_tile_y = boxes_per_tile_y

        self.box_gap_x = (self.max_value_x - self.min_value_x) / (self.boxes_per_tile_x - 1)
        self.box_gap_y = (self.max_value_y - self.min_value_y) / (self.boxes_per_tile_y - 1)

        self.tile_gap_x = self.box_gap_x / self.num_tiles
        self.tile_gap_y = self.box_gap_y / self.num_tiles

        self.tiles_start_x = np.zeros ((self.num_tiles, self.boxes_per_tile_x), dtype=float)
        self.tiles_start_y = np.zeros ((self.num_tiles, self.boxes_per_tile_y), dtype=float)

        offset_x = min_value_x
        offset_y = min_value_y

        for tile in range (self.num_tiles) :
            current_x = offset_x
            current_y = offset_y

            for box in range (self.boxes_per_tile_x) :
                self.tiles_start_x[tile][box] = current_x
                current_x += self.box_gap_x

            for box in range (self.boxes_per_tile_y) :
                self.tiles_start_y[tile][box] = current_y
                current_y += self.box_gap_y

            offset_x -= self.tile_gap_x
            offset_y -= self.tile_gap_y

    def coded_representation (self, x, y) :
        state = []
        

        for tile in range (self.num_tiles) :
            box_x = 0
            box_y = 0
            
            for box in range (self.boxes_per_tile_x) :
                if box == self.boxes_per_tile_x - 1 :
                    box_x = self.boxes_per_tile_x - 1
                    break

                if x >= self.tiles_start_x[tile][box] and x < self.tiles_start_x[tile][box+1] :
                    box_x = box
                    break

            for box in range (self.boxes_per_tile_y) :
                if box == self.boxes_per_tile_y - 1 :
                    box_y = self.boxes_per_tile_y - 1
                    break

                if y >= self.tiles_start_y[tile][box] and y < self.tiles_start_y[tile][box+1] :
                    box_y = box
                    break
            
            state.append (tile * self.boxes_per_tile_x * self.boxes_per_tile_y + box_x * self.boxes_per_tile_x + box_y)

        return state