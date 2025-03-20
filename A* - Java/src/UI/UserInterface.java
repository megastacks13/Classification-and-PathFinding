package UI;

import Structure.Table;

public interface UserInterface {

    void display(int[][] table);
    int askForGreaterValueThan(int n, String msg);
    void askInput(Table tbl);
    void showPathNotFound();
}
