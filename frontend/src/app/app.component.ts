import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { PredictorComponent } from "./components/predictor/predictor.component";

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, PredictorComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'frontend';
}
