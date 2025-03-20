import Structure.NodeManager;
import Structure.Table;
import UI.ConsoleUserInterface;
import UI.UserInterface;

public class Main {

    public static void main(String[] args) {
        UserInterface ui = new ConsoleUserInterface();
        int rows = 0;
        int columns = 0;

        rows = ui.askForGreaterValueThan(2, "Introduce número de filas (>2)");
        columns = ui.askForGreaterValueThan(2, "Introduce número de columnas (>2)");
        Table tbl = new Table(rows, columns);
        ui.askInput(tbl);
        if (!tbl.executeAlgorithm()) ui.showPathNotFound();
        else ui.display(tbl.getTable());
    }
}
